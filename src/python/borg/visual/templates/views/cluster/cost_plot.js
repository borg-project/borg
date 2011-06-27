//
// CLUSTER VIEW COST PLOT
//

bv.views.cluster.plots.cost = {
    title: "Runtime Distribution",
    order: 1
};

bv.views.cluster.plots.cost.create = function(view, nodes) {
    var plot = Object.create(bv.views.cluster.plots.cost);

    plot.view = view;
    plot.nodes = nodes;
    plot.solver = view.resources.solvers[0];
    plot.maxCost = d3.max(view.resources.runs, function(d) {
        return d3.max(d.runs, function(e) { return e.cost; });
    });

    return plot.initialize();
};

bv.views.cluster.plots.cost.initialize = function() {
    // mise en place
    var this_ = this;
    var $this = $(this);

    // control elements
    this.nodes.solverSelect =
        $('<p>Solver:<select id="bv-plot-solver-select"></select></p>')
            .appendTo(this.nodes.controlsDiv)
            .children("select")
            .get(0);

    d3.select(this.nodes.solverSelect)
        .selectAll("option")
        .data(this.view.resources.solvers)
        .enter()
        .append("option")
        .attr("value", function(d) { return d; })
        .text(function(d) { return d; });

    $(this.nodes.solverSelect)
        .selectmenu({style: "dropdown"})
        .change(function(event) {
            this_.solver = $(this).val(); 

            this_.update();
        });

    // prepare for plotting
    this.chart = 
        bv.barChart.create(
            {
                chartDiv: this.nodes.chartDiv,
                chartSVG: this.nodes.chartSVG
            },
            {
                xAxis: "Time to Solution (CPU s)",
                yAxis: "Fraction of Runs"
            },
            40
        );

    $(this.view.selections).bind("changed", function() { this_.update(); });

    this.update();

    return this;
};

bv.views.cluster.plots.cost.digitize = function(values, count, left, right, normalize) {
    // initialize histogram
    var histogram = {
        left: left,
        right: right,
        width: (right - left) / count,
        bins: [],
        ranges: [],
        values: []
    };

    for(var i = 0; i < count; i += 1) {
        histogram.bins[i] = 0;
        histogram.values[i] = [];
        histogram.ranges[i] = {
            left: i * histogram.width,
            right: (i + 1) * histogram.width
        };
    }

    values.forEach(function(value) {
        if(value > right) {
            throw {message: "value out of histogram range"};
        }

        var i = Math.min(count - 1, Math.floor(value / histogram.width));

        histogram.bins[i] += 1;
        histogram.values[i].push(value);
    });

    // optionally normalize counts to densities
    if(normalize) {
        var max = histogram.bins.reduce(function(a, b) { return a + b; });

        if(max !== 0.0) {
            for(var i = 0; i < histogram.bins.length; i += 1) {
                histogram.bins[i] /= max;
            }
        }
    }

    // ...
    return histogram;
};

bv.views.cluster.plots.cost.update = function() {
    var this_ = this;
    var xDomain = [0, this.maxCost];
    var yDomain = [0, 0];
    var xLabels = [];
    var allSeries = [];

    this.view.selections.get().forEach(function(selection) {
        if(selection.visible && selection.ready) {
            var values = [];

            selection.instances().forEach(function(instance) {
                instance.runs.forEach(function(run) {
                    if(run.solver === this_.solver) {
                        values.push({
                            run: run,
                            instance: instance,
                            valueOf: function() { return run.cost; }
                        });
                    }
                });
            });

            var histogram = this_.digitize(values, this_.chart.nbars, 0, this_.maxCost, true);

            if(xLabels.length === 0) {
                histogram.ranges.forEach(function(range) {
                    xLabels.push("%.0f--%.0f".format(range.left, range.right));
                });
            }

            allSeries.push({
                id: selection.number,
                color: selection.color,
                bars: histogram.bins.map(function(density, i) {
                    if(density > yDomain[1]) {
                        yDomain[1] = density;
                    }

                    var points = histogram.values[i].map(function(v) { return v.instance.node; });
                    var bar = {
                        height: density,
                        left: histogram.ranges[i].left,
                        right: histogram.ranges[i].right,
                        associated: points
                    };

                    $(bar)
                        .bind("highlighted", function() {
                            d3.selectAll(points).classed("highlighted", true);
                        })
                        .bind("unhighlighted", function() {
                            d3.selectAll(points).classed("highlighted", false);
                        });

                    return bar;
                })
            });
        }
    });

    this.chart.update(allSeries, xDomain, yDomain, xLabels);
};

bv.views.cluster.plots.cost.destroy = function() {
    $(this.view.selections).unbind("changed", this.update);
    $(this.nodes.controlsDiv).empty();

    this.chart.destroy();
};

