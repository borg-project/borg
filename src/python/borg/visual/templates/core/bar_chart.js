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
    var d3chart = d3.select(this.nodes.chartSVG);

    this.nodes.ticksGroup = d3chart.append("svg:g").node();
    this.nodes.barsGroup = d3chart.append("svg:g").node();
    this.nodes.yLabelsGroup = d3chart.append("svg:g").node();

    // add labels to axes
    this.nodes.xLabel =
        d3chart
            .append("svg:text")
            .attr("x", 50)
            .attr("y", 2)
            .attr("dy", "1em")
            .style("font-weight", "bold")
            .text(this.labels.xAxis)
            .node();

    d3chart
        .append("svg:g")
        .attr(
            "transform",
            "translate(%s, %s) rotate(-90)".format(
                14,
                $(this.nodes.chartDiv).innerHeight() - 4
            )
        )
        .append("svg:text")
        .style("font-weight", "bold")
        .text(this.labels.yAxis);

    return this;
};

bv.barChart.update = function(allSeries, xDomain, yDomain, xLabels) {
    // prepare scales
    var $chartDiv = $(this.nodes.chartDiv);
    var xScale = d3.scale.linear().domain(xDomain).rangeRound([52, $chartDiv.innerWidth()]);
    var yScale = d3.scale.linear().domain(yDomain).rangeRound([$chartDiv.innerHeight(), 18]);

    // update
    this.updateTicks(xScale, yScale);
    this.updateBars(allSeries, xScale, yScale, xLabels);
};

bv.barChart.updateTicks = function(xScale, yScale) {
    // y-axis ticks
    var yTicks = yScale.ticks(8);
    var d3yTicks = d3.select(this.nodes.ticksGroup).selectAll(".tick-line").data(yTicks);

    d3yTicks
        .enter()
        .append("svg:line")
        .attr("class", "tick-line")
        .attr("x1", xScale.range()[0] - 30)
        .attr("y1", yScale.range()[0])
        .attr("x2", xScale.range()[1])
        .attr("y2", yScale.range()[0])
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
        .attr("y1", yScale.range()[0])
        .attr("y2", yScale.range()[0])
        .remove();

    // y-axis labels
    var d3yLabels =
        d3.select(this.nodes.yLabelsGroup)
            .selectAll("text")
            .data(yTicks);

    d3yLabels
        .enter()
        .append("svg:text")
        .attr("x", xScale.range()[0] - 4)
        .attr("y", yScale.range()[0] + 20)
        .style("text-anchor", "end")
        .text(function(d) { return "%.2f".format(d); })
        .transition()
        .duration(500)
        .attr("y", function(d) { return yScale(d) - 1; });
    d3yLabels
        .transition()
        .duration(500)
        .attr("y", function(d) { return yScale(d) - 1; })
        .text(function(d) { return "%.2f".format(d); });
    d3yLabels
        .exit()
        .transition()
        .duration(500)
        .attr("y", yScale.range()[0] + 20)
        .remove();
};

bv.barChart.updateBars = function(allSeries, xScale, yScale, xLabels) {
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
        .attr("class", "series")
        .style("fill", function(d) { return d.color; })
        .style("stroke", function(d) { return d.color; })
        .selectAll()
        .data(function(d) { return d.bars; })
        .enter()
        .append("svg:rect")
        .attr("x", function(d) { return xScale(d.left) - 0.5; })
        .attr("y", function(d) { return yScale.range()[0]; })
        .attr("width", function(d) { return xScale(d.right) - xScale(d.left) - 2; })
        .attr("height", 0)
        .each(function(d, i) {
            // prepare event handlers
            var bar = this;
            var dxLabel = d3.select(this_.nodes.xLabel);

            this.highlight = function() {
                d3.select(bar).classed("highlighted", true);

                dxLabel.text("%s: %s".format(this_.labels.xAxis, xLabels[i]));

                $(bar.__data__).trigger("highlighted");
            };
            this.unhighlight = function() {
                d3.select(bar).classed("highlighted", false);

                dxLabel.text(this_.labels.xAxis);

                $(bar.__data__).trigger("unhighlighted");
            };
            this.reassociate = function(associated) {
                if(this.deassociate !== undefined) {
                    this.deassociate();
                }

                this.deassociate = function() {
                    $(associated)
                        .unbind("mouseover", this.highlight)
                        .unbind("mouseout", this.unhighlight);
                };

                $(associated)
                    .bind("mouseover", this.highlight)
                    .bind("mouseout", this.unhighlight);
            };

            // and bind them
            $(this)
                .bind("mouseover", this.highlight)
                .bind("mouseout", this.unhighlight);

            this.reassociate(d.associated);
        })
        .transition()
        .duration(500)
        .attr("y", function(d) { return yScale(d.height) + 0.5; })
        .attr("height", function(d) { return yScale.range()[0] - yScale(d.height); });
    d3allSeries
        .selectAll("rect")
        .data(function(d) { return d.bars; })
        .each(function(d) { this.reassociate(d.associated); })
        .transition()
        .duration(500)
        .attr("y", function(d) { return yScale(d.height) + 0.5; })
        .attr("height", function(d) { return yScale.range()[0] - yScale(d.height); });
    d3allSeries
        .exit()
        .selectAll("rect")
        .each(function(d) { this.deassociate(); });
    d3allSeries
        .exit()
        .transition()
        .duration(500)
        .call(function(s) { d3.select(s).remove(); })
        .selectAll("rect")
        .attr("y", function(d) { return yScale.range()[0]; })
        .attr("height", 0);

    d3.select(this.nodes.barsGroup)
        .selectAll("g.series")
        .sort(function(a, b) {
            var sum = function(x, y) { return x + y; };
            var height = function(x) { return Math.pow(x.height, 2); };

            var aSum = a.bars.map(height).reduce(sum);
            var bSum = b.bars.map(height).reduce(sum);

            return d3.descending(aSum, bSum);
        });
};

bv.barChart.destroy = function() {
    d3.select(this.nodes.barsGroup)
        .selectAll("g.series")
        .selectAll("rect")
        .each(function(d) { this.deassociate(); })
        .remove();

    $(this.nodes.chartSVG).empty();
};

