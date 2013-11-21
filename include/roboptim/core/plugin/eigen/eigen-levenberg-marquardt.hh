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

# include <boost/mpl/vector.hpp>

# include <roboptim/core/solver.hh>
# include <roboptim/core/sum-of-c1-squares.hh>

#include <unsupported/Eigen/NonLinearOptimization>

namespace roboptim {
  namespace eigen {
    /// \brief Solver implementing a variant of Levenberg-Marquardt algorithm.
    ///
    /// This solver tries to minimize the euclidean norm of a vector valued
    /// function.
    class SolverWithJacobian :
      public Solver<SumOfC1Squares, boost::mpl::vector<> >
    {
    public:
      /// \brief Parent type
      typedef Solver<SumOfC1Squares, boost::mpl::vector<> >
      parent_t;
      /// \brief Cost function type
      typedef problem_t::function_t function_t;
      /// \brief type of result
      typedef parent_t::result_t result_t;
      /// \brief type of gradient
      typedef DifferentiableFunction::gradient_t gradient_t;
      /// \brief Size type
      typedef Function::size_type size_type;

      /// \brief Constructot by problem
      explicit SolverWithJacobian (const problem_t& problem);
      virtual ~SolverWithJacobian () throw ();
      /// \brief Solve the optimization problem
      virtual void solve () throw ();

      /// Number of variables
      size_type n () const
      {
	return n_;
      }

      /// Number of functions
      size_type m () const
      {
	return m_;
      }

      /// Get parameter
      Function::argument_t& parameter ()
      {
	return x_;
      }

      const Function::argument_t& parameter () const
      {
	return x_;
      }

      /// Get cost
      boost::shared_ptr <const DifferentiableFunction> cost () const
      {
        return cost_;
      }

    private:
      /// Number of variables
      size_type n_;
      /// Dimension of the cost function
      size_type m_;

      /// Parameter of the function
      Function::argument_t x_;

      /// Reference to cost function
      boost::shared_ptr <const DifferentiableFunction> cost_;

      /// Map of <optimization status, warning messages>
      std::map<Eigen::LevenbergMarquardtSpace::Status,std::string> warning_map_;
    }; // class SolverWithJacobian
  } // namespace eigen
} // namespace roboptim
#endif // ROBOPTIM_CORE_PLUGIN_EIGEN_EIGEN_LEVENBERG_MARQUARDT_HH
