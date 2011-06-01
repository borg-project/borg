//
// CLUSTER VIEW COST PLOT
//

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

    this.nodes.plot = dplot.node();

    dplot.append("svg:g");
    dplot.append("svg:g");

    var dlabels = dplot.append("svg:g");

    dlabels
        .append("svg:text")
        .attr("x", 4)
        .attr("y", 2)
        .attr("dy", "1em")
        .text("Time to Solution (CPU s) -->");

    dlabels
        .append("svg:g")
        .attr(
            "transform",
            "translate(%s, %s) rotate(90)".format(
                $(this.nodes.plot).innerWidth() - 10,
                20
            )
        )
        .append("svg:text")
        .text("Fraction of Runs -->");

    dlabels
        .append("svg:text")
        .attr("id", "bar-label")
        .attr("class", "borg-tooltip")
        .attr("filter", "url(#soft-glow)")
        .style("font-weight", "bold")
        .text("");

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

        for(var i = 0; i < histogram.bins.length; i += 1) {
            histogram.bins[i] /= max;
        }
    }

    // ...
    return histogram;
};

bv.views.cluster.plots.cost.update = function() {
    // recompute histograms
    var this_ = this;
    var series = [];
    var nbins = 40;

    this.view.selections.get().forEach(function(selection) {
        if(selection.visible) {
            var instances = selection.instances();
            var values = [];

            instances.forEach(function(instance) {
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

            series.push({
                selection: selection,
                histogram: this_.digitize(values, nbins, 0, this_.maxCost, true)
            });
        }
    });

    // update ticks and tick lines
    var $densityPlot = $(this.nodes.plot);
    var dticksGroup = d3.select("#density-plot > g:nth-of-type(1)");
    var maxBin = d3.max(series.map(function(d) { return d3.max(d.histogram.bins); }));
    var xScale = d3.scale.linear().domain([0, this_.maxCost]).rangeRound([0, $densityPlot.innerWidth() - 14]);
    var yScale = d3.scale.linear().domain([0, maxBin]).rangeRound([18, $densityPlot.innerHeight()]);
    var xTicks = [];

    for(var i = 0; i < nbins + 1; i += 1) {
        xTicks[i] = i * (this_.maxCost / nbins);
    }

    var dxticks = dticksGroup.selectAll(".tick").data(xTicks);
    var dyticks = dticksGroup.selectAll(".tick-line").data(yScale.ticks(8));

    dxticks
        .enter()
        .append("svg:line")
        .classed("tick", true)
        .attr("x1", function(d) { return xScale(d) + 0.5; })
        .attr("y1", yScale(0))
        .attr("x2", function(d) { return xScale(d) + 0.5; })
        .attr("y2", yScale(0) + 10);

    dyticks
        .enter()
        .append("svg:line")
        .classed("tick-line", true)
        .attr("x1", xScale.range()[0])
        .attr("y1", yScale.range()[1])
        .attr("x2", xScale.range()[1])
        .attr("y2", yScale.range()[1])
        .transition()
        .duration(500)
        .attr("y1", function(d) { return yScale(d) + 0.5; })
        .attr("y2", function(d) { return yScale(d) + 0.5; });
    dyticks
        .transition()
        .duration(500)
        .attr("y1", function(d) { return yScale(d) + 0.5; })
        .attr("y2", function(d) { return yScale(d) + 0.5; });
    dyticks
        .exit()
        .transition()
        .duration(500)
        .attr("y1", yScale.range()[1])
        .attr("y2", yScale.range()[1])
        .remove();

    // update histogram bars
    var getRectData = function(d) {
        return d.histogram.bins.map(function(density, i) {
            return {
                density: density,
                range: d.histogram.ranges[i],
                values: d.histogram.values[i],
                points: d.histogram.values[i].map(function(v) { return v.instance.node; })
            };
        });
    };
    var dSeries =
        d3.select("#density-plot > g:nth-of-type(2)")
            .selectAll("g.series")
            .data(series, function(d) { return d.selection.number; });

    dSeries
        .enter()
        .append("svg:g")
        .classed("series", true)
        .style("fill", function(d) {
            if(d.selection === null) {
                return "#aaaaaa";
            }
            else {
                return d.selection.color;
            }
        })
        .each(function(d) {
            var dself = d3.select(this);

            $(d.selection).bind("highlight", function(e, state) {
                dself.classed("highlighted", state);
            });
        })
        .selectAll("rect")
        .data(getRectData)
        .enter()
        .append("svg:rect")
        .attr("x", function(d) { return xScale(d.range.left); })
        .attr("y", function(d) { return yScale.range()[1]; })
        .attr("width", function(d) { return xScale(d.range.right) - xScale(d.range.left); })
        .attr("height", 0)
        .each(function(d) {
            // prepare event handlers
            var dthis = d3.select(this);
            var dlabel = d3.select("#bar-label");
            var dpoints = d3.selectAll(d.points);

            d.highlight = function() {
                var right = dthis.attr("x") > $densityPlot.innerWidth() / 2;

                dthis.classed("highlighted", true);
                dpoints.classed("highlighted", true);

                dlabel
                    .attr("x", xScale(right ? d.range.right : d.range.left))
                    .attr("y", yScale.range()[0] + 18)
                    .attr("text-anchor", right ? "end" : "start")
                    .text("%.0f--%.0f".format(d.range.left, d.range.right));
            };
            d.unhighlight = function() {
                dthis.classed("highlighted", false);
                dpoints.classed("highlighted", false);

                dlabel.text("");
            };

            // and bind them
            $(this)
                .bind("mouseover", d.highlight)
                .bind("mouseout", d.unhighlight);
            $(d.points)
                .bind("mouseover", d.highlight)
                .bind("mouseout", d.unhighlight);
        })
        .transition()
        .duration(500)
        .attr("y", function(d) {
            return yScale.range()[1] + yScale.range()[0] - yScale(d.density);
        })
        .attr("height", function(d) { return yScale(d.density); });
    dSeries
        .selectAll("rect")
        .data(getRectData)
        .transition()
        .duration(500)
        .attr("y", function(d) {
            return yScale.range()[1] + yScale.range()[0] - yScale(d.density);
        })
        .attr("height", function(d) { return yScale(d.density); });
    dSeries
        .exit()
        .each(function(d) {
            $(d.points)
                .unbind("mouseover", d.highlight)
                .unbind("mouseout", d.unhighlight);
        })
        .transition()
        .duration(500)
        .remove()
        .selectAll("rect")
        .attr("y", function(d) { return yScale.range()[1]; })
        .attr("height", 0);
};

bv.views.cluster.plots.cost.destroy = function() {
    $(this.view.selections).unbind("changed", this.update);
};

