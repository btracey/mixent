// Copyright ©2016 The Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// mixent implements routines for estimating the entropy of a mixture distribution.
// See
//  Estimating Mixture Entropy using Pairwise Distances by A. Kolchinksy and
//  B. Tracey
// for more information.
// Documentation notation: the mixture distribution is
//  p(x) = \sum_{i=1}^N w_i p_i(x)
// The symbol X will be used to represent the mixture distribution p(x), and
// X_i will represent p_i(x), that is the distribution of the i^th mixture component.
// It will also use \sum_i as a shorthand for \sum_{i=1}^N.
//
// Please note that some of the types implemented here work for all mixture
// distributions, while others only work for special cases, such as mixture of
// Gaussians.
package mixent

import (
	"math"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"
	"github.com/gonum/stat/distmv"
)

var (
	dimensionMismatch = "mixent: component dimension mismatch"
)

// Component is an individual component in a mixture distribution.
type Component interface {
	Entropy() float64
}

// Estimator is a type that can estimate the entropy of a mixture of Components.
type Estimator interface {
	MixtureEntropy(components []Component, weights []float64) float64
}

// Distancer is a type that can compute the distance between two components.
// The distance returned must be greater than zero, and must equal zero if a and
// b are identical distributions.
type Distancer interface {
	Distance(a, b Component) float64
}

// DistNormaler is a type that can compute the distance between two Normal distributions.
type DistNormaler interface {
	DistNormal(l, r *distmv.Normal) float64
}

// NormalDistance wraps a DistNormaler for use with components.
type NormalDistance struct {
	DistNormaler
}

// Distance computes the distance between two Normal components. Distance panics
// if the underlying type of the components is not *distmv.Normal or if the
// dimensions are unequal.
func (n NormalDistance) Distance(l, r Component) float64 {
	l1 := l.(*distmv.Normal)
	l2 := r.(*distmv.Normal)
	return n.DistNormal(l1, l2)
}

// DistNormaler is a type that can compute the distance between two Normal distributions.
type DistUniformer interface {
	DistUniform(l, r *distmv.Uniform) float64
}

// NormalDistance wraps a DistUniformer for use with components.
type UniformDistance struct {
	DistUniformer
}

// Distance computes the distance between two Normal components. Distance panics
// if the underlying type of the components is not *distmv.Uniform or if the
// dimensions are unequal.
func (u UniformDistance) Distance(l, r Component) float64 {
	l1 := l.(*distmv.Uniform)
	l2 := r.(*distmv.Uniform)
	return u.DistUniform(l1, l2)
}

// LowerGaussian is a lower bound on the entropy of a mixture of Gaussians.
// See the MixtureEntropy method for more information.
type LowerGaussian struct{}

// MixtureEntropy computes a lower bound on the entropy of a mixture of Gaussians.
// It implements Section V.A, Theorem 2 of
//  Huber, Marco F., et al. "On entropy approximation for Gaussian mixture
//  random vectors." Multisensor Fusion and Integration for Intelligent Systems,
//  2008. MFI 2008. IEEE International Conference on. IEEE, 2008.
// The entropy estimate is
//  H(x) ≈ -\sum_i w_i log(\sum_j w_j z_{i,j})
//  z_{i,j} = 𝒩(μ_i; μ_j, Σ_i+Σ_j)
// All of the mixture components must be *distmv.Normal and must have the same
// input dimension or MixtureEntropy will panic.
//
// If weights is nil, all components are assumed to have the same mixture entropy.
func (l LowerGaussian) MixtureEntropy(components []Component, weights []float64) float64 {
	n := len(components)

	// Convert the Component to *distmv.Normal and check the dimensions.
	norms := make([]*distmv.Normal, n)
	norms[0] = components[0].(*distmv.Normal)
	dim := norms[0].Dim()
	for i := 1; i < n; i++ {
		norm := components[i].(*distmv.Normal)
		if norm.Dim() != dim {
			panic(dimensionMismatch)
		}
		norms[i] = norm
	}
	if weights == nil {
		weights = make([]float64, n)
		for i := range weights {
			weights[i] = 1 / float64(n)
		}
	}

	// Create memory storage.
	mui := make([]float64, dim)
	muj := make([]float64, dim)
	covi := mat64.NewSymDense(dim, nil)
	covj := mat64.NewSymDense(dim, nil)
	covSum := mat64.NewSymDense(dim, nil)
	inner := make([]float64, n)

	// Estimate entropy.
	var ent float64
	for i, a := range norms {
		a.Mean(mui)
		a.CovarianceMatrix(covi)
		for j, b := range norms {
			b.Mean(muj)
			b.CovarianceMatrix(covj)
			covSum.AddSym(covi, covj)
			norm, ok := distmv.NewNormal(muj, covSum, nil)
			if !ok {
				// This should be impossible because the sum of PSD matrices is PSD.
				panic("mixent: covariance sum not positive definite")
			}
			inner[j] = norm.LogProb(mui) + math.Log(weights[j])
		}
		ent -= weights[i] * floats.LogSumExp(inner)
	}
	return ent
}

// JointEntropy provides an upper bound on the mixture entropy using
// the joint entropy of the mixture and the weights. See the MixtureEntropy
// method for more information.
type JointEntropy struct{}

// MixtureEntropy computes an upper bound to the mixture entropy for arbitrary
// mixture components.
//
// The joint entropy between the mixture distribution and the weight distribution
// is an upper bound on the entropy of the distribution
//  H(X) ≤ H(X, W) = H(X|W) + H(W) = \sum_i w_i H(p_i) + H(W)
//
// If weights is nil, all components are assumed to have the same mixture entropy.
func (JointEntropy) MixtureEntropy(components []Component, weights []float64) float64 {
	avgEnt := AvgEnt{}.MixtureEntropy(components, weights)
	if weights == nil {
		return avgEnt + math.Log(float64(len(components)))
	}
	avgEnt += stat.Entropy(weights)
	return avgEnt
}

// AvgEnt provides a lower bound on the mixture entropy using the average entropy
// of the components. See the MixtureEntropy method for more information.
type AvgEnt struct{}

// MixtureEntropy computes the entropy of the mixture conditional on the mixture
// weight, which is a lower bound to the true entropy of the mixture distribution.
//  H(X) ≥ H(X|W) = \sum_i w_i H(X_i)
// i.e. the average entropy of the components.
//
// If weights is nil, all components are assumed to have the same mixture entropy.
func (AvgEnt) MixtureEntropy(components []Component, weights []float64) float64 {
	if weights == nil {
		var avgEnt float64
		nf := float64(len(components))
		for _, v := range components {
			avgEnt += v.Entropy() / nf
		}
		return avgEnt
	}
	var avgEnt float64
	for i, v := range components {
		avgEnt += weights[i] * v.Entropy()
	}
	return avgEnt
}

// PairwiseDistance estimates the entropy using a pairwise distance metric.
// See the MixtureEntropy method for more information.
type PairwiseDistance struct {
	Distancer Distancer
}

// MixtureEntropy estimates the entropy using a distance metric between each pair
// of distributions. It implements EQUATION IN PAPER, that is
//  H(x) ≈ \sum_i w_i H(X_i) - \sum_i w_i ln(\sum_j w_j exp(-D(X_i || X_j)))
// As shown in
//  Estimating Mixture Entropy using Pairwise Distances by A. Kolchinksy and
//  B. Tracey
// this estimator has several nice properties (see the paper for the necessary
// conditions on D for the following to hold).
//
// 1) The estimate returned is always larger than Conditional and smaller than
// JointMixtureWeight. These two estimators differ by H(weights), so this provides
// a bound on the error.
//
// 2) The pairwise estimator becomes an exact estimate of the entropy as the mixture
// components become clustered. The number of clusters is arbitrary, so this estimator
// is exact when all mixture components are the same, all are very far apart from
// one another, or clustered into k far apart clusters.
//
// 3) The Bhattacharrya distance metric is a lower-bound on the mixture entropy
// and the Kullback-Leibler divergence is an upper bound on the mixture entropy.
// Any distance metric D_B ≤ D ≤ D_KL will provide an estimate in between these
// two distances.
//
// If weights is nil, all components are assumed to have the same mixture entropy.
func (p PairwiseDistance) MixtureEntropy(components []Component, weights []float64) float64 {
	n := len(components)
	if weights == nil {
		weights = make([]float64, n)
		for i := range weights {
			weights[i] = 1 / float64(n)
		}
	}
	var avgEntropy float64
	for i, v := range components {
		avgEntropy += weights[i] * v.Entropy()
	}

	var outer float64
	inner := make([]float64, n) // Storing the sums for numeric stability
	for i, xi := range components {
		for j, xj := range components {
			inner[j] = math.Log(weights[j]) - p.Distancer.Distance(xi, xj)
		}
		outer += weights[i] * floats.LogSumExp(inner)
	}
	return avgEntropy - outer
}

// ComponentCenters estimates the entropy based on the probability at the component
// centers.
type ComponentCenters struct{}

// MixtureEntropy computes the estimate of the entropy based on the average
// probability of the cluster centers.
//  H(x) ≈ - \sum_i w_i ln \sum_j w_j p_j(μ_i)
// If weights is nil, all components are assumed to have the same mixture entropy.
//
// Currently only coded for Gaussian components.
func (p ComponentCenters) MixtureEntropy(components []Component, weights []float64) float64 {
	n := len(components)
	norms := make([]*distmv.Normal, n)
	norms[0] = components[0].(*distmv.Normal)
	dim := norms[0].Dim()
	for i := 1; i < n; i++ {
		norm := components[i].(*distmv.Normal)
		if norm.Dim() != dim {
			panic(dimensionMismatch)
		}
		norms[i] = norm
	}
	if weights == nil {
		weights = make([]float64, n)
		for i := range weights {
			weights[i] = 1 / float64(n)
		}
	}

	// Compute center probabilities
	inner := make([]float64, n)
	var ent float64
	mean := make([]float64, dim)
	for i, norm := range norms {
		norm.Mean(mean)
		for j, norm2 := range norms {
			inner[j] = math.Log(weights[j]) + norm2.LogProb(mean)
		}
		ent += weights[i] * floats.LogSumExp(inner)
	}
	return -ent
}
