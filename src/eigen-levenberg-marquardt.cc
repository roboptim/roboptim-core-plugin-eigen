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

#include <boost/assign/list_of.hpp>

#include <roboptim/core/function.hh>
#include <roboptim/core/problem.hh>
#include <roboptim/core/solver-error.hh>
#include <roboptim/core/plugin/eigen/eigen-levenberg-marquardt.hh>

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
	: Functor<double> (solver.problem().function ().baseFunction ()->inputSize (),
	                   solver.problem().function ().baseFunction ()->outputSize ()),
	  solver_ (solver)
      {
      }

      int operator() (const FunctorType::InputType& x, FunctorType::ValueType& fvec) const
      {
        fvec = (*solver_.cost ())(x);

        return 0;
      }

      int df (const FunctorType::InputType& x, FunctorType::ValueType& jac_row, VectorXd::Index rownb)
      {
	::roboptim::Function::size_type row =
          static_cast< ::roboptim::Function::size_type> (rownb) - 2;
        jac_row = solver_.cost ()->gradient (x, row);

        return 0;
      }

      int df (const FunctorType::InputType& x, FunctorType::JacobianType& jac)
      {
	solver_.parameter () = x;
        jac = solver_.cost ()->jacobian (x);

        return 0;
      }

      /// RobOptim solver
      S& solver_;
    };


    SolverWithJacobian::SolverWithJacobian (const problem_t& problem) :
      Solver <SumOfC1Squares, boost::mpl::vector<> >
      (problem),
      n_ (problem.function ().baseFunction ()->inputSize ()),
      m_ (problem.function ().baseFunction ()->outputSize ()),
      x_ (n_),
      cost_ (problem.function ().baseFunction ())
    {
      // Initialize this class parameters
      x_.setZero ();

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
         "computation.");
    }

    SolverWithJacobian::~SolverWithJacobian () throw ()
    {
    }

    // Utility macro to print result with warning message
#define LOAD_RESULT_WARNINGS(STATUS)                                    \
    case STATUS:                                                        \
    {                                                                   \
      ResultWithWarnings result (n_, 1);                                \
      result.x = x_;                                                    \
      result.value = problem ().function () (result.x);                 \
      result.warnings.push_back (SolverWarning (warning_map_[STATUS])); \
      result_ = result;                                                 \
    }                                                                   \
    break;


    void SolverWithJacobian::solve () throw ()
    {
      // Load optional starting point
      if (problem ().startingPoint ())
        {
          x_ = *(problem ().startingPoint ());
        }

      using namespace Eigen::LevenbergMarquardtSpace;
      solver_functor<SolverWithJacobian> functor (*this);
      LevenbergMarquardt<solver_functor<SolverWithJacobian> > lm (functor);
      Status info = lm.minimize (x_);

      switch (info)
        {
        case LevenbergMarquardtSpace::ImproperInputParameters:
          result_ = SolverError ("Improper input parameters");
          break;

        LOAD_RESULT_WARNINGS (RelativeReductionTooSmall)
        LOAD_RESULT_WARNINGS (RelativeErrorTooSmall)
        LOAD_RESULT_WARNINGS (RelativeErrorAndReductionTooSmall)
        LOAD_RESULT_WARNINGS (CosinusTooSmall)
        LOAD_RESULT_WARNINGS (TooManyFunctionEvaluation)
        LOAD_RESULT_WARNINGS (FtolTooSmall)
        LOAD_RESULT_WARNINGS (XtolTooSmall)
        LOAD_RESULT_WARNINGS (GtolTooSmall)
        LOAD_RESULT_WARNINGS (UserAsked)

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

  ROBOPTIM_DLLEXPORT unsigned getSizeOfProblem ();
  ROBOPTIM_DLLEXPORT const char* getTypeIdOfConstraintsList ();
  ROBOPTIM_DLLEXPORT solver_t* create (const SolverWithJacobian::problem_t& pb);
  ROBOPTIM_DLLEXPORT void destroy (solver_t* p);

  ROBOPTIM_DLLEXPORT unsigned getSizeOfProblem ()
  {
    return sizeof (solver_t::problem_t);
  }

  ROBOPTIM_DLLEXPORT const char* getTypeIdOfConstraintsList ()
  {
    return typeid (solver_t::problem_t::constraintsList_t).name ();
  }

  ROBOPTIM_DLLEXPORT solver_t* create (const SolverWithJacobian::problem_t& pb)
  {
    return new SolverWithJacobian (pb);
  }

  ROBOPTIM_DLLEXPORT void destroy (solver_t* p)
  {
    delete p;
  }
}
