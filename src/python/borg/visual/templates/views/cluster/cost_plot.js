//
// CLUSTER VIEW COST PLOT
//

//var maxBar = d3.max(allSeries, function(d) { return d3.max(d.bars, function(b) { return b.height; }); });

bv.views.cluster.plots.cost = {};

bv.views.cluster.plots.cost.create = function(view, controlsNode) {
    var plot = Object.create(bv.views.cluster.plots.cost);

    plot.view = view;
    plot.nodes = {
        controls: controlsNode
    };
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
    this.nodes.density =
        $('<p>Histogrammed:</p>')
            .appendTo(this.nodes.controls)
            .append("<select></select>")
            .children()
            .attr("id", "density-select")
            .get(0);

    d3.select(this.nodes.density)
        .selectAll("option")
        .data(this.view.resources.solvers)
        .enter()
        .append("option")
        .attr("value", function(d) { return d; })
        .text(function(d) { return d; });

    $(this.nodes.density)
        .selectmenu({ style: "dropdown" })
        .change(function(event) {
            this_.solver = $(this_.nodes.density).val(); 

            this_.update();
        });

    // prepare for plotting
    var dplot =
        d3.select("body")
            .append("div")
            .attr("id", "density-plot-area")
            .append("svg:svg")
            .attr("id", "density-plot");

    this.chart = 
        bv.barChart.create(
            {chart: dplot.node()},
            {x_axis: "Time to Solution (CPU s) -->", y_axis: "Fraction of Runs -->"},
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
    var allSeries = [];

    this.view.selections.get().forEach(function(selection) {
        if(selection.visible) {
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

            allSeries.push({
                id: selection.number,
                color: selection.color,
                bars: histogram.bins.map(function(density, i) {
                    if(density > yDomain[1]) {
                        yDomain[1] = density;
                    }

                    return {
                        height: density,
                        left: histogram.ranges[i].left,
                        right: histogram.ranges[i].right,
                        proxy: null // XXX
                    };
                })
            });
        }
    });

    this.chart.update(allSeries, xDomain, yDomain);
};

bv.views.cluster.plots.cost.destroy = function() {
    $(this.view.selections).unbind("changed", this.update);
};

