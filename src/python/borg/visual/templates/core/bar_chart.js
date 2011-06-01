//
// CLUSTER VIEW BAR CHART (BASE)
//

bv.barChart = {};

bv.barChart.create = function(nodes, labels, nbars) {
    var chart = Object.create(bv.barChart);

    chart.nodes = nodes;
    chart.labels = labels;
    chart.nbars = nbars;

    return chart.initialize();
};

bv.barChart.initialize = function() {
    // prepare for plotting
    var d3chart = d3.select(this.nodes.chart);

    this.nodes.ticksGroup = d3chart.append("svg:g").node();
    this.nodes.barsGroup = d3chart.append("svg:g").node();

    // add axes labels
    var d3labels = d3chart.append("svg:g");

    d3labels
        .append("svg:text")
        .attr("x", 4)
        .attr("y", 2)
        .attr("dy", "1em")
        .text(this.labels.x_axis);
    d3labels
        .append("svg:g")
        .attr(
            "transform",
            "translate(%s, %s) rotate(90)".format(
                $(this.nodes.chart).innerWidth() - 10,
                20
            )
        )
        .append("svg:text")
        .text(this.labels.y_axis);

    return this;
};

bv.barChart.update = function(allSeries, xDomain, yDomain) {
    // prepare scales
    var $chart = $(this.nodes.chart);
    var xScale = d3.scale.linear().domain(xDomain).rangeRound([0, $chart.innerWidth() - 14]);
    var yScale = d3.scale.linear().domain(yDomain).rangeRound([18, $chart.innerHeight()]);

    // update
    this.updateTicks(xScale, yScale);
    this.updateBars(allSeries, xScale, yScale);
};

bv.barChart.updateTicks = function(xScale, yScale) {
    // x-axis ticks
    var xTicks = [];

    for(var i = 0; i < this.nbars + 1; i += 1) {
        xTicks[i] = i * (xScale.domain()[1] - xScale.domain()[0]) / this.nbars;
    }

    var d3ticksGroup = d3.select(this.nodes.ticksGroup);
    var d3xTicks = d3ticksGroup.selectAll(".tick").data(xTicks);

    d3xTicks
        .enter()
        .append("svg:line")
        .classed("tick", true)
        .attr("x1", function(d) { return xScale(d) + 0.5; })
        .attr("y1", yScale.range()[0])
        .attr("x2", function(d) { return xScale(d) + 0.5; })
        .attr("y2", yScale.range()[0] + 10);

    // y-axis ticks
    var d3yTicks = d3ticksGroup.selectAll(".tick-line").data(yScale.ticks(8));

    d3yTicks
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
    d3yTicks
        .transition()
        .duration(500)
        .attr("y1", function(d) { return yScale(d) + 0.5; })
        .attr("y2", function(d) { return yScale(d) + 0.5; });
    d3yTicks
        .exit()
        .transition()
        .duration(500)
        .attr("y1", yScale.range()[1])
        .attr("y2", yScale.range()[1])
        .remove();
};

bv.barChart.updateBars = function(allSeries, xScale, yScale) {
    // mise en place
    var this_ = this;

    // update bars
    var d3allSeries =
        d3.select(this.nodes.barsGroup)
            .selectAll("g.series")
            .data(allSeries, function(d) { return d.id; });

    d3allSeries
        .enter()
        .append("svg:g")
        .classed("series", true)
        .style("fill", function(d) { return d.color; })
        .each(function(d) {
            var d3this = d3.select(this);

            $(d.proxy).bind("highlight", function(e, state) {
                d3this.classed("highlighted", state);
            });
        })
        .selectAll()
        .data(function(d) { return d.bars; })
        .enter()
        .append("svg:rect")
        .attr("x", function(d) { return xScale(d.left); })
        .attr("y", function(d) { return yScale.range()[1]; })
        .attr("width", function(d) { return xScale(d.right) - xScale(d.left); })
        .attr("height", 0)
        //.each(function(d) {
            //// prepare event handlers
            //var dthis = d3.select(this);
            //var dlabel = d3.select("#bar-label");
            //var dpoints = d3.selectAll(d.points);

            //d.highlight = function() {
                //var right = dthis.attr("x") > $densityPlot.innerWidth() / 2;

                //dthis.classed("highlighted", true);
                //dpoints.classed("highlighted", true);

                //dlabel
                    //.attr("x", xScale(right ? d.range.right : d.range.left))
                    //.attr("y", yScale.range()[0] + 18)
                    //.attr("text-anchor", right ? "end" : "start")
                    //.text("%.0f--%.0f".format(d.range.left, d.range.right));
            //};
            //d.unhighlight = function() {
                //dthis.classed("highlighted", false);
                //dpoints.classed("highlighted", false);

                //dlabel.text("");
            //};

            //// and bind them
            //$(this)
                //.bind("mouseover", d.highlight)
                //.bind("mouseout", d.unhighlight);
            //$(d.points)
                //.bind("mouseover", d.highlight)
                //.bind("mouseout", d.unhighlight);
        //})
        .transition()
        .duration(500)
        .attr("y", function(d) {
            return yScale.range()[1] + yScale.range()[0] - yScale(d.height);
        })
        .attr("height", function(d) { return yScale(d.height); });
    d3allSeries
        .selectAll("rect")
        .data(function(d) { return d.bars; })
        .transition()
        .duration(500)
        .attr("y", function(d) {
            return yScale.range()[1] + yScale.range()[0] - yScale(d.height);
        })
        .attr("height", function(d) { return yScale(d.height); });
    d3allSeries
        .exit()
        //.each(function(d) {
            //$(d.points)
                //.unbind("mouseover", d.highlight)
                //.unbind("mouseout", d.unhighlight);
        //})
        .transition()
        .duration(500)
        .remove()
        .selectAll("rect")
        .attr("y", function(d) { return yScale.range()[1]; })
        .attr("height", 0);

    // update labels
};

bv.barChart.destroy = function() {
    // XXX
};

