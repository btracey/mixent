// Copyright ©2016 The Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"image/color"
	"log"
	"math"
	"math/rand"

	"github.com/btracey/mixent"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
	"gonum.org/v1/gonum/stat/distuv"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

func main() {
	names := []string{
		"gauss_sweep_skew",
		"gauss_sweep_size",
		"gauss_sweep_clusters",
		"gauss_sweep_dim",

		"unif_sweep_size",
		"unif_sweep_skew",
		"unif_sweep_clusters",
		"unif_sweep_dim",
	}

	plot.DefaultFont = "Helvetica"

	for _, name := range names {
		fmt.Println("Running ", name)
		run := GetRun(name)

		// Fix the random samples for each run. Keep the randomness consistent
		// across the hyper sweep to reduce noise.
		rnd := rand.New(rand.NewSource(1))

		// The code allows the dimension of the problem to adjust with the
		// hyperparameters. Find the maximum dimension used.
		var maxDim int
		for _, hyper := range run.Hypers {
			dim := run.DistGen.CompDim(hyper)
			if dim > maxDim {
				maxDim = dim
			}
		}

		// Generate random samples for computing the MC entropy.
		randComps := make([]int, run.MCEntropySamples)
		for i := range randComps {
			randComps[i] = rnd.Intn(run.NumComponents)
		}
		mcSamps := mat.NewDense(run.MCEntropySamples, maxDim, nil)
		for i := 0; i < run.MCEntropySamples; i++ {
			for j := 0; j < maxDim; j++ {
				mcSamps.Set(i, j, rnd.Float64())
			}
		}

		// Generate the random numbers for components.
		nRand := run.DistGen.NumRandom()
		compSamps := mat.NewDense(run.NumComponents, nRand, nil)
		for i := 0; i < run.NumComponents; i++ {
			for j := 0; j < nRand; j++ {
				compSamps.Set(i, j, rnd.Float64())
			}
		}

		// Sweep over the hyperparameter, and estimate the entropy with all of the
		// estimators
		entComponents := make([]mixent.Component, run.NumComponents)
		components := make([]Component, run.NumComponents)
		mcEnt := make([]float64, len(run.Hypers)) // store of the entropy from Monte Carlo.

		estEnts := mat.NewDense(len(run.Hypers), len(run.Estimators), nil) // entropy from estimators.
		for i, hyper := range run.Hypers {
			fmt.Println(name, i, "of", len(run.Hypers))

			// Construct the components given the random samples and hyperparameter.
			for j := range components {
				components[j] = run.DistGen.ComponentFrom(compSamps.RawRowView(j), hyper)
				entComponents[j] = components[j]
			}

			// Estimate the entropy with all the estimators.
			for j, estimator := range run.Estimators {
				v := estimator.MixtureEntropy(entComponents, nil)
				estEnts.Set(i, j, v)
			}

			// Estimate the entropy from Monte Carlo.
			dim := run.DistGen.CompDim(hyper)
			sv := mcSamps.Slice(0, run.MCEntropySamples, 0, dim).(*mat.Dense)
			mcEnt[i] = mcEntropy(components, randComps, sv)
		}

		// Plot the results.
		err := makePlots(run, mcEnt, estEnts)
		if err != nil {
			log.Fatal(err)
		}
	}
}

// mcEntropy estimates the entropy of the mixture distribution given the pre-drawn
// random components and samples.
func mcEntropy(components []Component, randComps []int, mcSamps *mat.Dense) float64 {
	nSamp, dim := mcSamps.Dims()
	if len(randComps) != nSamp {
		panic("rand mismatch")
	}
	var ent float64
	lw := -math.Log(float64(len(components))) // probability of chosing component.
	x := make([]float64, dim)
	lps := make([]float64, len(components))
	for i := range randComps {
		// Extract the sampled x location.
		comp := randComps[i]
		components[comp].Quantile(x, mcSamps.RawRowView(i))
		// Compute \sum_i w_i p(x_i).
		for j := range lps {
			lps[j] = components[j].LogProb(x) + lw
		}
		ent += floats.LogSumExp(lps)
	}
	return -ent / float64(nSamp)
}

type Run struct {
	Name             string
	DistGen          DistributionGenerator
	MCEntropySamples int
	NumComponents    int

	Hypers     []float64
	Estimators []mixent.Estimator

	XLabel string
	LogX   bool
}

func GetRun(name string) Run {
	var isUniform bool

	c := Run{
		Name:          name,
		NumComponents: 100,
	}

	switch name {
	default:
		panic("unknown case name")
	case "gauss_sweep_size":
		dim := 10
		hypers := make([]float64, 30)
		floats.Span(hypers, -2.5, 1.5)
		for i, v := range hypers {
			hypers[i] = math.Exp(v)
		}

		c.DistGen = ShiftCenters{dim, false}
		c.Hypers = hypers
		c.XLabel = "Ln(Mean Spread)"
		c.LogX = true
	case "gauss_sweep_skew":
		dim := 10
		hypers := make([]float64, 30)
		floats.Span(hypers, 0, 7)
		for i, v := range hypers {
			hypers[i] = math.Exp(v)
		}

		c.DistGen = GaussianFixedCenter{dim}
		c.Hypers = hypers
		c.XLabel = "Ln(Covariance Similarity)"
		c.LogX = true
	case "gauss_sweep_clusters":
		dim := 10
		hypers := make([]float64, 30)
		floats.Span(hypers, -2.5, 0.5)
		for i, v := range hypers {
			hypers[i] = math.Exp(v)
		}

		// Generate centers from uniform ball
		rnd := rand.New(rand.NewSource(1))
		nClusters := 5
		centers := mat.NewDense(nClusters, dim, nil)
		for i := 0; i < nClusters; i++ {
			for j := 0; j < dim; j++ {
				centers.Set(i, j, rnd.NormFloat64())
			}
		}

		c.DistGen = ClusterCenters{centers, false}
		c.Hypers = hypers
		c.XLabel = "Ln(Cluster Spread)"
		c.LogX = true
	case "gauss_sweep_dim":
		hypers := make([]float64, 8)
		floats.Span(hypers, 1, 25)
		maxDim := int(hypers[len(hypers)-1])

		c.DistGen = FlexibleDim{maxDim, false}
		c.Hypers = hypers
		c.XLabel = "Dimension"
		c.LogX = false
	case "unif_sweep_size":
		isUniform = true
		dim := 10
		hypers := make([]float64, 30)
		floats.Span(hypers, -4, 1.5)
		for i, v := range hypers {
			hypers[i] = math.Exp(v)
		}

		c.DistGen = ShiftCenters{dim, true}
		c.Hypers = hypers
		c.XLabel = "Ln(Mean Spread)"
		c.LogX = true
	case "unif_sweep_clusters":
		rnd := rand.New(rand.NewSource(1))
		isUniform = true
		dim := 10
		hypers := make([]float64, 30)
		floats.Span(hypers, -6, 0)
		for i, v := range hypers {
			hypers[i] = math.Exp(v)
		}

		// Geterate centers from uniform ball
		nClusters := 5
		centers := mat.NewDense(nClusters, dim, nil)
		for i := 0; i < nClusters; i++ {
			for j := 0; j < dim; j++ {
				centers.Set(i, j, rnd.NormFloat64())
			}
		}

		c.DistGen = ClusterCenters{centers, true}
		c.Hypers = hypers
		c.XLabel = "Ln(Cluster Spread)"
		c.LogX = true
	case "unif_sweep_dim":
		isUniform = true
		hypers := make([]float64, 8)
		floats.Span(hypers, 1, 16)
		maxDim := int(hypers[len(hypers)-1])

		c.DistGen = FlexibleDim{maxDim, true}
		c.Hypers = hypers
		c.XLabel = "Dimension"
		c.LogX = false
	case "unif_sweep_skew":
		isUniform = true
		dim := 10
		hypers := make([]float64, 50)
		floats.Span(hypers, -3.5, 3)
		for i, v := range hypers {
			hypers[i] = math.Exp(v)
		}

		c.DistGen = UniformFixedCenter{dim}
		c.Hypers = hypers
		c.XLabel = "Ln(Width Variability)"
		c.LogX = true
	}
	if isUniform {
		c.Estimators = []mixent.Estimator{
			mixent.JointEntropy{},
			mixent.AvgEnt{},
			mixent.PairwiseDistance{mixent.UniformDistance{distmv.KullbackLeibler{}}},
			mixent.PairwiseDistance{mixent.UniformDistance{distmv.Bhattacharyya{}}},
			mixent.ComponentCenters{},
			mixent.ELK{},
		}
		c.MCEntropySamples = 5000
	} else {
		c.Estimators = []mixent.Estimator{
			mixent.JointEntropy{},
			mixent.AvgEnt{},
			mixent.PairwiseDistance{mixent.NormalDistance{distmv.KullbackLeibler{}}},
			mixent.PairwiseDistance{mixent.NormalDistance{distmv.Bhattacharyya{}}},
			mixent.ComponentCenters{},
			mixent.ELK{},
		}
		c.MCEntropySamples = 2000
	}
	return c
}

// Component is the same as a mixent.Component, except can also compute probabilities
// and quantiles
type Component interface {
	mixent.Component
	distmv.Quantiler
	distmv.LogProber
}

// DistributionGenerator is a type for generating the Components of a mixture model.
type DistributionGenerator interface {
	// NumRandom is the amount of random numbers needed to generate a component
	// (random location of the mean, etc.)
	NumRandom() int
	// ComponentFrom takes a set of random numbers and turns it into a component
	ComponentFrom(rands []float64, hyper float64) Component
	// CompDim returns the dimension as a function of the hyperparameter.
	CompDim(hyper float64) int
}

// ShiftCenters generates fixed-width components with center locations randomly
// generated on the hypercube with size hyper.
type ShiftCenters struct {
	Dim     int
	Uniform bool // if true, Uniform components. If false, Gaussian.
}

func (n ShiftCenters) CompDim(hyper float64) int {
	return n.Dim
}

func (n ShiftCenters) NumRandom() int {
	return n.Dim
}

func (n ShiftCenters) ComponentFrom(rands []float64, hyper float64) Component {
	if len(rands) != n.Dim {
		panic("bad ")
	}
	mean := make([]float64, len(rands))
	for i := range mean {
		mean[i] = distuv.Normal{Mu: 0, Sigma: hyper}.Quantile(rands[i])
	}
	if n.Uniform {
		bounds := make([]distmv.Bound, n.Dim)
		for i := range bounds {
			bounds[i].Min = mean[i] - 1
			bounds[i].Max = mean[i] + 1
		}
		return distmv.NewUniform(bounds, nil)
	}
	cov := mat.NewSymDense(n.Dim, nil)
	for i := 0; i < n.Dim; i++ {
		cov.SetSym(i, i, 1)
	}
	norm, ok := distmv.NewNormal(mean, cov, nil)
	if !ok {
		panic("bad covariance")
	}
	return norm
}

// ClusterCenters generates components according to their cluster. The hyperparameter
// controls the spread of the cluster locations.
type ClusterCenters struct {
	Centers *mat.Dense
	Uniform bool // if true, Uniform components. If false, Gaussian.
}

func (n ClusterCenters) CompDim(hyper float64) int {
	_, dim := n.Centers.Dims()
	return dim
}

func (n ClusterCenters) NumRandom() int {
	// Which cluster
	return 1
}

func (n ClusterCenters) ComponentFrom(rands []float64, hyper float64) Component {
	nClust, dim := n.Centers.Dims()
	// Get the number of the cluster
	v := rands[0]
	v *= float64(nClust)
	idx := int(math.Floor(v))

	mean := make([]float64, dim)
	copy(mean, n.Centers.RawRowView(idx))

	for i := range mean {
		mean[i] *= hyper
	}
	if n.Uniform {
		bounds := make([]distmv.Bound, len(mean))
		for i := range bounds {
			bounds[i].Min = mean[i] - 1
			bounds[i].Max = mean[i] + 1
		}
		return distmv.NewUniform(bounds, nil)
	}
	cov := mat.NewSymDense(dim, nil)
	for i := 0; i < dim; i++ {
		cov.SetSym(i, i, 1)
	}
	norm, ok := distmv.NewNormal(mean, cov, nil)
	if !ok {
		panic("bad covariance")
	}
	return norm
}

// GaussianFixedCenter generates Gaussian components with a fixed center location
// and covariance drawn from a Wishart distribution. As the hyperparameter increases,
// the covariances tend to the identity matrix.
type GaussianFixedCenter struct {
	Dim int
}

func (w GaussianFixedCenter) CompDim(hyper float64) int {
	return w.Dim
}

func (w GaussianFixedCenter) NumRandom() int {
	// Center location + chi^2 variables + normal variables
	return w.Dim + w.Dim + (w.Dim-1)*w.Dim/2
}

func (w GaussianFixedCenter) ComponentFrom(rands []float64, hyper float64) Component {
	dim := w.Dim
	nu := float64(dim) + hyper

	// The v matrix is Sigma_0^-1, and Sigma_0 = nu*I. So v = 1/nu * I. We are
	// computing the cholesky decomposition, so the square root of that
	cholv := mat.NewTriDense(dim, mat.Upper, nil)
	for i := 0; i < dim; i++ {
		cholv.SetTri(i, i, math.Sqrt(1/nu))
	}

	// The center is just fixed, the first rands.
	mu := make([]float64, w.Dim)
	for i := 0; i < dim; i++ {
		mu[i] = distuv.Normal{Mu: 0, Sigma: 1}.Quantile(rands[i])
	}
	rands = rands[dim:]

	u := mat.NewTriDense(dim, mat.Upper, nil)
	// Get the chi^2 random variables. Diagonal generated from Chi^2 of nu-i
	// degrees of freedom.
	for i := 0; i < dim; i++ {
		v := distuv.ChiSquared{
			K:   nu - float64(i),
			Src: nil,
		}.Quantile(rands[i])
		u.SetTri(i, i, math.Sqrt(v))
	}

	// Get the normal random variables for the off-diagonal
	rands = rands[dim:]
	for i := 0; i < dim; i++ {
		for j := i + 1; j < dim; j++ {
			u.SetTri(i, j, distuv.UnitNormal.Quantile(rands[0]))
			rands = rands[1:]
		}
	}

	var t mat.TriDense
	t.MulTri(u, cholv)

	var c mat.Cholesky
	c.SetFromU(&t)

	var cov mat.SymDense
	c.ToSym(&cov)

	// TODO(btracey): Can set directly from Cholesky.
	norm, ok := distmv.NewNormal(mu, &cov, nil)
	if !ok {
		panic("bad norm")
	}
	return norm
}

// UniformFixedCenter generates Uniform components with a fixed center location
// and widths drawn from a Gamma distribution. As the hyperparameter increases,
// the widths tend to unity.
type UniformFixedCenter struct {
	Dim int
}

func (u UniformFixedCenter) CompDim(hyper float64) int {
	return u.Dim
}

func (u UniformFixedCenter) NumRandom() int {
	// Center location + Gamma quantile.
	return u.Dim + 1
}

func (u UniformFixedCenter) ComponentFrom(rands []float64, hyper float64) Component {
	bounds := make([]distmv.Bound, u.Dim)
	// width is last random variable
	width := distuv.Gamma{Alpha: 1 + hyper, Beta: 1 + hyper}.Quantile(rands[u.Dim])
	for i := 0; i < u.Dim; i++ {
		mean := distuv.Normal{Mu: 0, Sigma: 1}.Quantile(rands[i])
		bounds[i].Min = mean - width
		bounds[i].Max = mean + width
	}
	return distmv.NewUniform(bounds, nil)
}

// FlexibleDim generates components whose dimension increases with the hyperparameter.
type FlexibleDim struct {
	MaxDim  int
	Uniform bool // if true, Uniform components. If false, Gaussian.
}

func (f FlexibleDim) CompDim(hyper float64) int {
	return int(hyper)
}

func (f FlexibleDim) NumRandom() int {
	return f.MaxDim
}

func (f FlexibleDim) ComponentFrom(rands []float64, hyper float64) Component {
	dim := f.CompDim(hyper)

	// Center is the first dim entries
	mean := make([]float64, dim)
	for i := 0; i < dim; i++ {
		mean[i] = distuv.Normal{Mu: 0, Sigma: 1}.Quantile(rands[i])
	}

	if f.Uniform {
		bounds := make([]distmv.Bound, len(mean))
		for i := range bounds {
			bounds[i].Min = (mean[i] - 1)
			bounds[i].Max = (mean[i] + 1)
		}
		return distmv.NewUniform(bounds, nil)
	}

	sigma := eyeSym(dim)
	n, ok := distmv.NewNormal(mean, sigma, nil)
	if !ok {
		panic("bad covariance")
	}
	return n
}

func eyeSym(dim int) *mat.SymDense {
	m := mat.NewSymDense(dim, nil)
	for i := 0; i < dim; i++ {
		m.SetSym(i, i, 1)
	}
	return m
}

func makePlots(run Run, mcEnt []float64, estEnts *mat.Dense) error {
	// Collect all of the data into lines
	x := make([]float64, len(run.Hypers))
	copy(x, run.Hypers)
	if run.LogX {
		for i, v := range x {
			x[i] = math.Log(v)
		}
	}
	minEnt := floats.Min(mcEnt)
	maxEnt := floats.Max(mcEnt)

	pts := make(plotter.XYs, len(x))
	for i := range pts {
		pts[i].X = x[i]
		pts[i].Y = mcEnt[i]
	}

	l, err := plotter.NewLine(pts)
	if err != nil {
		return err
	}
	linedata := []interface{}{"Monte Carlo", l}

	for j, estimator := range run.Estimators {
		y := mat.Col(nil, j, estEnts)
		min := floats.Min(y)
		if min < minEnt {
			minEnt = min
		}
		max := floats.Max(y)
		if max > maxEnt {
			maxEnt = max
		}

		if _, ok := estimator.(mixent.JointEntropy); ok {
			continue
		}
		if _, ok := estimator.(mixent.AvgEnt); ok {
			continue
		}

		pts2 := make(plotter.XYs, len(x))
		for i := range pts2 {
			pts2[i].X = x[i]
			pts2[i].Y = y[i]
		}
		l, err := plotter.NewLine(pts2)
		if err != nil {
			return err
		}
		name, ls := plotMapName(estimator)
		l.LineStyle = ls
		linedata = append(linedata, name)
		linedata = append(linedata, l)
	}

	p, err := plot.New()
	if err != nil {
		return err
	}

	// Draw a polygon with the HuberUpper and AvgEnt
	polyXYs := make(plotter.XYs, len(x)*2)
	for i, est := range run.Estimators {
		_, ok := est.(mixent.JointEntropy)
		if ok {
			for j, v := range x {
				polyXYs[j].X = v
				polyXYs[j].Y = estEnts.At(j, i)
			}
		}
	}
	for i, est := range run.Estimators {
		_, ok := est.(mixent.AvgEnt)
		if ok {
			for j, v := range x {
				polyXYs[2*len(x)-j-1].X = v
				polyXYs[2*len(x)-j-1].Y = estEnts.At(j, i)
			}
		}
	}
	poly, err := plotter.NewPolygon(polyXYs)
	if err != nil {
		log.Fatal(err)
	}
	polyColor := color.RGBA{242, 251, 222, 255}
	poly.Color = polyColor
	poly.LineStyle.Color = polyColor
	p.Add(poly)

	makeLegend := run.Name == "gauss_sweep_dim" || run.Name == "unif_sweep_dim"

	for i := 0; i < len(linedata); i += 2 {
		p.Add(linedata[i+1].(plot.Plotter))
		if makeLegend {
			p.Legend.Add(linedata[i].(string), linedata[i+1].(plot.Thumbnailer))
		}
	}
	if makeLegend {
		p.Legend.Add("[H(X|C), H(X,C)]", poly)
		p.Legend.Top = true
		p.Legend.Left = true
	}

	p.X.Label.Text = run.XLabel
	p.Y.Label.Text = "Entropy (nats)"

	p.Y.Max = (maxEnt-minEnt)*1.05 + minEnt
	p.Y.Min = maxEnt + (minEnt-maxEnt)*1.05

	err = p.Save(5*vg.Inch, 3*vg.Inch, "plot_"+run.Name+".pdf")
	return err
}

func plotMapName(e mixent.Estimator) (string, draw.LineStyle) {
	ls := plotter.DefaultLineStyle
	switch t := e.(type) {
	default:
		panic("unknown type")
	case mixent.JointEntropy:
		ls.Color = plotutil.Color(1)
		ls.Dashes = plotutil.Dashes(1)
		return "Huber Upper", ls
	case mixent.ELK:
		ls.Color = plotutil.Color(2)
		ls.Dashes = plotutil.Dashes(2)
		return "Exp. Lik. Ker.", ls
	case mixent.AvgEnt:
		ls.Color = plotutil.Color(3)
		ls.Dashes = plotutil.Dashes(3)
		return "Avg. Ent.", ls
	case mixent.ComponentCenters:
		ls.Color = plotutil.Color(4)
		ls.Dashes = plotutil.Dashes(4)
		return "KDE", ls
	case mixent.PairwiseDistance:
		var inner interface{}
		switch v := t.Distancer.(type) {
		case mixent.NormalDistance:
			inner = v.DistNormaler
		case mixent.UniformDistance:
			inner = v.DistUniformer
		}
		switch inner.(type) {
		default:
			panic("unknown type")
		case distmv.KullbackLeibler:
			ls.Color = plotutil.Color(5)
			ls.Dashes = plotutil.Dashes(5)
			return "Pair Dist: KL", ls
		case distmv.Bhattacharyya:
			ls.Color = plotutil.Color(6)
			ls.Dashes = plotutil.Dashes(6)
			return "Pair Dist: Bhat.", ls
		case mixent.ELKDist:
			ls.Color = plotutil.Color(7)
			ls.Dashes = plotutil.Dashes(7)
			return "Pair Dist: ELK", ls
		}
	}
}
