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

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distmv"
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

// unifLogVolOverlap computes the log of the volume of the hyper-rectangle where
// both uniform distributions have positive probability.
func unifLogVolOverlap(b1, b2 []distmv.Bound) float64 {
	var logVolOverlap float64
	for dim, v1 := range b1 {
		v2 := b2[dim]
		// If the surfaces don't overlap, then the volume is 0
		if v1.Max <= v2.Min || v2.Max <= v1.Min {
			return math.Inf(-1)
		}
		vol := math.Min(v1.Max, v2.Max) - math.Max(v1.Min, v2.Min)
		logVolOverlap += math.Log(vol)
	}
	return logVolOverlap
}

// ELK implements the Expected Likelihood Kernel.
//  ELX(X_i, X_j) = \int_x p_i(x) p_j(x) dx
// The Expected Likelihood Kernel can be used to find a lower bound on the mixture
// entropy. See the Mixture Entropy method for more information
type ELK struct{}

// KernelNormal computes the log of the Expected Likelihood Kernel for two Gaussians.
//  ELK = 𝒩(μ_i; μ_j, Σ_i+Σ_j)
func (ELK) LogKernelNormal(l, r *distmv.Normal) float64 {
	if l.Dim() != r.Dim() {
		panic("mixent: normal dimension mismatch")
	}
	covl := l.CovarianceMatrix(nil)
	covr := r.CovarianceMatrix(nil)
	covSum := mat.NewSymDense(l.Dim(), nil)
	covSum.AddSym(covl, covr)
	norm, ok := distmv.NewNormal(r.Mean(nil), covSum, nil)
	if !ok {
		// This should be impossible because the sum of PSD matrices is PSD.
		panic("mixent: covariance sum not positive definite")
	}
	return norm.LogProb(l.Mean(nil))
}

func (ELK) LogKernelUniform(l, r *distmv.Uniform) float64 {
	// ELK is \int_x p_i(x) p_j(x) dx. So the value is constant over the region
	// of overlap.
	// = volume * p(x) * q(x)
	// = log(vol) + log(p(x)) + log(q(x))
	bl := l.Bounds(nil)
	br := r.Bounds(nil)
	overlap := unifLogVolOverlap(bl, br)
	logPl := -l.Entropy()
	logPr := -r.Entropy()
	return overlap + logPl + logPr
}

// MixtureEntropy computes an estimate of the mixture entropy using the Expected
// Likelihood kernel. The lower bound is
//  H(x) ≈ -\sum_i w_i log(\sum_j w_j z_{i,j})
// Currently only works with components that are *distmv.Normal or *distmv.Uniform.
func (elk ELK) MixtureEntropy(components []Component, weights []float64) float64 {
	n := len(components)
	if weights == nil {
		weights = make([]float64, n)
		for i := range weights {
			weights[i] = 1 / float64(n)
		}
	}
	if len(weights) != len(components) {
		panic("mixent: length mismatch")
	}
	inner := make([]float64, n)
	var ent float64
	for i, a := range components {
		for j, b := range components {
			var logKernel float64
			switch t := a.(type) {
			case *distmv.Normal:
				logKernel = elk.LogKernelNormal(t, b.(*distmv.Normal))
			case *distmv.Uniform:
				logKernel = elk.LogKernelUniform(t, b.(*distmv.Uniform))
			}
			inner[j] = logKernel + math.Log(weights[j])
		}
		ent -= weights[i] * floats.LogSumExp(inner)
	}
	return ent
}

// ELKDist is a distance metric based on the expected likelihood distance.
// It is equal to
//  - ln(ELK(X_i, X_j)/sqrt(ELK(X_i,X_i)*ELK(X_j,X_j)))
type ELKDist struct{}

func (ELKDist) DistNormal(l, r *distmv.Normal) float64 {
	return -ELK{}.LogKernelNormal(l, r) + 0.5*ELK{}.LogKernelNormal(l, l) + 0.5*ELK{}.LogKernelNormal(r, r)
}

func (ELKDist) DistUniform(l, r *distmv.Uniform) float64 {
	return -ELK{}.LogKernelUniform(l, r) + 0.5*ELK{}.LogKernelUniform(l, l) + 0.5*ELK{}.LogKernelUniform(r, r)
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

func (ComponentCenters) gaussianCenters(components []Component) *mat.Dense {
	r := len(components)
	c := components[0].(*distmv.Normal).Dim()

	centers := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		centers.SetRow(i, components[i].(*distmv.Normal).Mean(nil))
	}
	return centers
}

func (ComponentCenters) uniformCenters(components []Component) *mat.Dense {
	r := len(components)
	c := components[0].(*distmv.Uniform).Dim()

	centers := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		b := components[i].(*distmv.Uniform).Bounds(nil)
		for j := range b {
			centers.Set(i, j, (b[j].Max+b[j].Min)/2)
		}
	}
	return centers
}

// MixtureEntropy computes the estimate of the entropy based on the average
// probability of the cluster centers.
//  H(x) ≈ - \sum_i w_i ln \sum_j w_j p_j(μ_i)
// If weights is nil, all components are assumed to have the same mixture entropy.
//
// Currently only coded for Gaussian and Uniform components.
func (comp ComponentCenters) MixtureEntropy(components []Component, weights []float64) float64 {
	var centers *mat.Dense
	switch components[0].(type) {
	default:
		panic("componentcenters: unknown mixture type")
	case *distmv.Normal:
		centers = comp.gaussianCenters(components)
	case *distmv.Uniform:
		centers = comp.uniformCenters(components)
	}

	n := len(components)
	if weights == nil {
		weights = make([]float64, n)
		for i := range weights {
			weights[i] = 1 / float64(n)
		}
	}
	if len(weights) != len(components) {
		panic("mixent: length of components does not match weights")
	}

	// Compute center probabilities
	var ent float64
	inner := make([]float64, n)
	for i := range components {
		center := centers.RawRowView(i)
		for j, comp := range components {
			lp := comp.(distmv.LogProber)
			inner[j] = math.Log(weights[j]) + lp.LogProb(center)
		}
		ent += weights[i] * floats.LogSumExp(inner)
	}
	return -ent
}
