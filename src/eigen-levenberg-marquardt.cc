// Copyright (c) 2013 CNRS
// Authors: Benjamin Chretien


// This file is part of roboptim-core-plugin-eigen
// roboptim-core-plugin-eigen is free software: you can redistribute it
// and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version.

// roboptim-core-plugin-eigen is distributed in the hope that it will be
// useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Lesser Public License for more details.  You should have
// received a copy of the GNU Lesser General Public License along with
// roboptim-core-plugin-eigen  If not, see
// <http://www.gnu.org/licenses/>.

#include <cstring>
#include <map>
#include <stdexcept>

#include <boost/assign/list_of.hpp>

#include <roboptim/core/function.hh>
#include <roboptim/core/problem.hh>
#include <roboptim/core/solver-error.hh>
#include <roboptim/core/sum-of-c1-squares.hh>

#include <roboptim/core/plugin/eigen/eigen-levenberg-marquardt.hh>
#include <roboptim/core/plugin/eigen/config.hh>

#include <unsupported/Eigen/NonLinearOptimization>

using namespace Eigen;

namespace roboptim
{
  namespace eigen
  {
    /// Generic functor (based on Eigen's test files)
    template<typename _Scalar, int NX=Dynamic, int NY=Dynamic>
    struct Functor
    {
      typedef _Scalar Scalar;
      enum {
	InputsAtCompileTime = NX,
	ValuesAtCompileTime = NY
      };
      typedef Matrix<Scalar,InputsAtCompileTime,1> InputType;
      typedef Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
      typedef Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;
      typedef GenericFunction<roboptim::EigenMatrixDense>::size_type SizeType;

      const SizeType m_inputs, m_values;

      Functor () : m_inputs (InputsAtCompileTime),
	           m_values (ValuesAtCompileTime)
      {}

      Functor (SizeType inputs, SizeType values) : m_inputs (inputs),
                                                   m_values (values)
      {}

      SizeType inputs () const { return m_inputs; }
      SizeType values () const { return m_values; }

      // you should define that in the subclass :
      // void operator() (const InputType& x, ValueType* v, JacobianType* _j=0) const;
    };

    /// Functor wrapping RobOptim's differentiable function for Eigen
    template <typename S>
    struct solver_functor : Functor<double>
    {
      typedef Functor<double> FunctorType;

      solver_functor (S& solver)
	: Functor<double> (solver.baseCost ()->inputSize (),
	                   solver.baseCost ()->outputSize ()),
	  solver_ (solver)
      {
      }

      int operator() (const FunctorType::InputType& x, FunctorType::ValueType& fvec) const
      {
        fvec = (*solver_.baseCost ())(x);

        return 0;
      }

      int df (const FunctorType::InputType& x, FunctorType::ValueType& jac_row, VectorXd::Index rownb)
      {
	::roboptim::Function::size_type row =
          static_cast< ::roboptim::Function::size_type> (rownb) - 2;
        jac_row = solver_.baseCost ()->gradient (x, row);

        return 0;
      }

      int df (const FunctorType::InputType& x, FunctorType::JacobianType& jac)
      {
	solver_.parameter () = x;
        jac = solver_.baseCost ()->jacobian (x);

        return 0;
      }

      /// RobOptim solver
      S& solver_;
    };


    SolverWithJacobian::SolverWithJacobian (const problem_t& problem) :
      parent_t (problem),
      baseCost_ (),
      n_ (),
      m_ (),
      x_ (),
      solverState_ (problem)
    {
      // Initialize this class parameters
      initialize (problem);
      initializeParameters();

      // Load <Status, warning message> map
      using namespace Eigen::LevenbergMarquardtSpace;
      warning_map_ = boost::assign::map_list_of
        (RelativeReductionTooSmall,
         "The cosine of the angle between fvec and any column of "
         "the jacobian is at most gtol in absolute value.")
        (RelativeErrorTooSmall,
         "Relative error too small.")
        (RelativeErrorAndReductionTooSmall,
         "Relative error and reduction too small.")
        (CosinusTooSmall,
         "The cosine of the angle between fvec and any column of "
         "the jacobian is at most gtol in absolute value.")
        (TooManyFunctionEvaluation,
         "Too many function evaluations done.")
        (FtolTooSmall,
         "ftol is too small. No further reduction in the sum of "
         "squares is possible")
        (XtolTooSmall,
         "xtol is too small. No further improvement in the "
         "approximate solution x is possible.")
        (GtolTooSmall,
         "gtol is too small. fvec is orthogonal to the columns of "
         "the jacobian to machine precision.")
        (UserAsked,
         "Error in user-implemented evaluation or gradient "
         "computation.")
        // C++11 workaround for Boost bug
        .convert_to_container<std::map<Eigen::LevenbergMarquardtSpace::Status,
                              std::string> >();
    }

    SolverWithJacobian::~SolverWithJacobian ()
    {
    }

    void SolverWithJacobian::initialize (const problem_t& pb)
    {
      const SumOfC1Squares* cost
        = dynamic_cast<const SumOfC1Squares*> (&pb.function ());

      if (!cost)
      {
        throw std::runtime_error ("the eigen-levenberg-marquardt plugin expects"
                                  " a SumOfC1Squares cost function");
      }

      baseCost_ = cost->baseFunction ();

      n_ = baseCost_->inputSize ();
      m_ = baseCost_->outputSize ();

      x_.resize (n_);
      x_.setZero ();
    }

#define DEFINE_PARAMETER(KEY, DESCRIPTION, VALUE)     \
  do                                                  \
  {                                                   \
    this->parameters_[KEY].description = DESCRIPTION; \
    this->parameters_[KEY].value = VALUE;             \
  } while (0)

    void SolverWithJacobian::initializeParameters()
    {
      this->parameters_.clear();

      DEFINE_PARAMETER("eigen.factor", "Sets the step bound for the diagonal shift", 100.);
      DEFINE_PARAMETER("eigen.maxfev", "Sets the maximum number of function evaluation", 400);
      DEFINE_PARAMETER("eigen.ftol", "Sets the tolerance for the norm of the vector function", std::sqrt(NumTraits<parent_t::problem_t::value_type>::epsilon()));
      DEFINE_PARAMETER("eigen.xtol", "Sets the tolerance for the norm of the solution vector", std::sqrt(NumTraits<parent_t::problem_t::value_type>::epsilon()));
      DEFINE_PARAMETER("eigen.gtol", "Sets the tolerance for the norm of the gradient of the error vector", 0.);
      DEFINE_PARAMETER("eigen.epsilon", "Sets the error precision", 0.);
    }

    template <typename U>
    Eigen::LevenbergMarquardtSpace::Status
    SolverWithJacobian::minimize (U& lm)
    {
      LevenbergMarquardtSpace::Status status = lm.minimizeInit (x_);
      if (status == LevenbergMarquardtSpace::ImproperInputParameters)
        return status;
      do {
        status = lm.minimizeOneStep (x_);
        if (!callback_.empty ())
          {
            solverState_.x() = x_;
            solverState_.cost () = lm.fnorm * lm.fnorm;
            callback_ (problem (), solverState_);
          }
      } while (status == LevenbergMarquardtSpace::Running);

      return status;
    }


    void SolverWithJacobian::solve ()
    {
      // Load optional starting point
      if (problem ().startingPoint ())
        {
          x_ = *(problem ().startingPoint ());
        }

      using namespace Eigen::LevenbergMarquardtSpace;
      solver_functor<SolverWithJacobian> functor (*this);
      LevenbergMarquardt<solver_functor<SolverWithJacobian> > lm (functor);

      // Custom parameters
      lm.parameters.factor = boost::get<double>(this->parameters_["eigen.factor"].value);
      lm.parameters.maxfev = boost::get<int>(this->parameters_["eigen.maxfev"].value);
      lm.parameters.ftol = boost::get<double>(this->parameters_["eigen.ftol"].value);
      lm.parameters.xtol = boost::get<double>(this->parameters_["eigen.xtol"].value);
      lm.parameters.gtol = boost::get<double>(this->parameters_["eigen.gtol"].value);
      lm.parameters.epsfcn = boost::get<double>(this->parameters_["eigen.epsilon"].value);

      Status info = minimize (lm);

      switch (info)
        {
        case LevenbergMarquardtSpace::ImproperInputParameters:
          result_ = SolverError ("Improper input parameters");
          break;

        // Warnings
        case RelativeReductionTooSmall:
        case RelativeErrorTooSmall:
        case RelativeErrorAndReductionTooSmall:
        case CosinusTooSmall:
        case TooManyFunctionEvaluation:
        case FtolTooSmall:
        case XtolTooSmall:
        case GtolTooSmall:
        case UserAsked:
        {
          ResultWithWarnings result (n_, 1);
          result.x = x_;
          result.value = problem ().function () (result.x);
          result.warnings.push_back (SolverWarning (warning_map_[info]));
          result_ = result;
        }
        break;

        default:
          result_ = SolverError ("Return value not documented");
        }
    }

  } // namespace eigen
} // end of namespace roboptim

extern "C"
{
  using namespace roboptim::eigen;
  typedef SolverWithJacobian::parent_t solver_t;

  ROBOPTIM_CORE_PLUGIN_EIGEN_DLLEXPORT unsigned getSizeOfProblem ();
  ROBOPTIM_CORE_PLUGIN_EIGEN_DLLEXPORT const char* getTypeIdOfConstraintsList ();
  ROBOPTIM_CORE_PLUGIN_EIGEN_DLLEXPORT solver_t* create (const SolverWithJacobian::problem_t& pb);
  ROBOPTIM_CORE_PLUGIN_EIGEN_DLLEXPORT void destroy (solver_t* p);

  ROBOPTIM_CORE_PLUGIN_EIGEN_DLLEXPORT unsigned getSizeOfProblem ()
  {
    return sizeof (solver_t::problem_t);
  }

  ROBOPTIM_CORE_PLUGIN_EIGEN_DLLEXPORT const char* getTypeIdOfConstraintsList ()
  {
    return typeid (solver_t::problem_t::constraintsList_t).name ();
  }

  ROBOPTIM_CORE_PLUGIN_EIGEN_DLLEXPORT solver_t* create (const SolverWithJacobian::problem_t& pb)
  {
    return new SolverWithJacobian (pb);
  }

  ROBOPTIM_CORE_PLUGIN_EIGEN_DLLEXPORT void destroy (solver_t* p)
  {
    delete p;
  }
}
