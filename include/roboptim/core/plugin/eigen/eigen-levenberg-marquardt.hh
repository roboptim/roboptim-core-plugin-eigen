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

#ifndef ROBOPTIM_CORE_PLUGIN_EIGEN_EIGEN_LEVENBERG_MARQUARDT_HH
# define ROBOPTIM_CORE_PLUGIN_EIGEN_EIGEN_LEVENBERG_MARQUARDT_HH

# include <map>
# include <string>

# include <boost/mpl/vector.hpp>

# include <roboptim/core/solver.hh>
# include <roboptim/core/solver-state.hh>

#include <unsupported/Eigen/NonLinearOptimization>

namespace roboptim {
  namespace eigen {
    /// \brief Solver implementing a variant of Levenberg-Marquardt algorithm.
    ///
    /// This solver tries to minimize the euclidean norm of a vector valued
    /// function.
    class SolverWithJacobian : public Solver<EigenMatrixDense>
    {
    public:
      /// \brief Parent type
      typedef Solver<EigenMatrixDense> parent_t;
      /// \brief Cost function type
      typedef problem_t::function_t function_t;
      /// \brief type of result
      typedef parent_t::result_t result_t;
      /// \brief type of gradient
      typedef DifferentiableFunction::gradient_t gradient_t;
      /// \brief Size type
      typedef Function::size_type size_type;

      /// \brief Solver state
      typedef SolverState<parent_t::problem_t> solverState_t;

      /// \brief RobOptim callback
      typedef parent_t::callback_t callback_t;

      /// \brief Constructot by problem
      explicit SolverWithJacobian (const problem_t& problem);
      virtual ~SolverWithJacobian ();

      /// \brief Solve the optimization problem
      virtual void solve ();

      /// \brief Return the number of variables.
      size_type n () const
      {
	return n_;
      }

      /// \brief Return the number of functions.
      size_type m () const
      {
	return m_;
      }

      /// \brief Get the optimization parameters.
      Function::argument_t& parameter ()
      {
	return x_;
      }

      /// \brief Get the optimization parameters.
      const Function::argument_t& parameter () const
      {
	return x_;
      }

      /// \brief Set the callback called at each iteration.
      virtual void
      setIterationCallback (callback_t callback)
      {
        callback_ = callback;
      }

      /// \brief Get the callback called at each iteration.
      const callback_t& callback () const
      {
        return callback_;
      }

      const boost::shared_ptr<const DifferentiableFunction> baseCost () const
      {
        return baseCost_;
      }

    private:
      /// \brief Minimize the cost function.
      template <typename U>
      Eigen::LevenbergMarquardtSpace::Status minimize (U& lm);

      /// \brief Initialize the solver.
      /// \param problem problem.
      void initialize (const problem_t& problem);
      void initializeParameters();

    private:
      /// \brief Base cost function.
      boost::shared_ptr<const DifferentiableFunction> baseCost_;

      /// \brief Number of variables
      size_type n_;
      /// \brief Dimension of the cost function
      size_type m_;

      /// \brief Parameter of the function
      Function::argument_t x_;

      /// \brief Map of <optimization status, warning messages>
      std::map<Eigen::LevenbergMarquardtSpace::Status,std::string> warning_map_;

      /// \brief State of the solver at each iteration
      solverState_t solverState_;

      /// \brief Intermediate callback (called at each end of iteration).
      callback_t callback_;
    }; // class SolverWithJacobian
  } // namespace eigen
} // namespace roboptim
#endif // ROBOPTIM_CORE_PLUGIN_EIGEN_EIGEN_LEVENBERG_MARQUARDT_HH
