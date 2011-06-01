//
// CLUSTER VIEW SUCCESS PLOT
//

//var maxBar = d3.max(allSeries, function(d) { return d3.max(d.bars, function(b) { return b.height; }); });

bv.views.cluster.plots.success = {};

bv.views.cluster.plots.success.create = function(view, nodes) {
    var plot = Object.create(bv.views.cluster.plots.success);

    plot.view = view;
    plot.nodes = nodes;
    plot.solvers = view.resources.solvers;

    return plot.initialize();
};

bv.views.cluster.plots.success.initialize = function() {
    // mise en place
    var this_ = this;
    var $this = $(this);

    // set up DOM
    var dplot =
        d3.select("body")
            .append("div")
            .attr("id", "density-plot-area")
            .append("svg:svg")
            .attr("id", "density-plot");

    this.chart = 
        bv.barChart.create(
            {chart: dplot.node()},
            {x_axis: "Solver Name", y_axis: "Probability of Success"},
            this.solvers.length
        );

    // general setup
    this.solverIndices = {};

    this.solvers.forEach(function(name, i) {
        this_.solverIndices[name] = i;
    });

    // start
    $(this.view.selections).bind("changed", function() { this_.update(); });

    this.update();

    return this;
};

bv.views.cluster.plots.success.update = function() {
    var this_ = this;
    var xDomain = [0, this.solvers.length];
    var yDomain = [0, null];
    var allSeries = [];

    this.view.selections.get().forEach(function(selection) {
        if(selection.visible) {
            var successes = this_.solvers.map(function() { return 0; });
            var attempts = this_.solvers.map(function() { return 0; });

            selection.instances().forEach(function(instance) {
                instance.runs.forEach(function(run) {
                    var i = this_.solverIndices[run.solver];

                    attempts[i] += 1;

                    if(run.answer !== null) {
                        successes[i] += 1;
                    }
                });
            });

            allSeries.push({
                id: selection.number,
                color: selection.color,
                bars: successes.map(function(count, i) {
                    var fraction = count / attempts[i];

                    if(yDomain[1] === null || fraction > yDomain[1]) {
                        yDomain[1] = fraction;
                    }

                    return {
                        height: fraction,
                        left: i,
                        right: i + 1,
                        proxy: null // XXX
                    };
                })
            });
        }
    });

    this.chart.update(allSeries, xDomain, yDomain);
};

bv.views.cluster.plots.success.destroy = function() {
    $(this.view.selections).unbind("changed", this.update);
};

