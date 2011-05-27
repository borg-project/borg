//
// CLUSTER VIEW
//

var makeHistogram = function(values, count, left, right, normalize) {
    // initialize histogram
    var histogram = {
        left: left,
        right: right,
        width: (right - left) / count,
        bins: [],
        ranges: []
    };

    for(var i = 0; i < count; i += 1) {
        histogram.bins[i] = 0;
        histogram.ranges[i] = {
            left: i * histogram.width,
            right: (i + 1) * histogram.width
        };
    }

    values.forEach(function(value) {
        if(value > right) {
            throw {message: "value out of histogram range"};
        }

        histogram.bins[Math.floor(value / histogram.width)] += 1;
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

var newClusterView = function(ui) {
    var view = newView([
        ui.resource("runs", "runs.json"),
        ui.resource("solvers", "solvers.json"),
        ui.resource("instances", "instances.json"),
        ui.resource("membership", "membership.json"),
        ui.resource("projection", "projection.json"),
    ]);

    view.updatePlot = function() {
        // recompute histograms
        var solver = $("#density-select").val();
        var series = [];
        var nbins = 40;

        this.selections.forEach(function(selection) {
            if(selection.visible) {
                var instances = selection.instances();
                var values = [];

                view.runs.forEach(function(runsOn) {
                    if(instances.indexOf(runsOn.instance) >= 0) {
                        runsOn.runs.forEach(function(run) {
                            if(run.solver === solver) {
                                values.push(run.cost);
                            }
                        });
                    }
                });

                series.push({
                    selection: selection,
                    histogram: makeHistogram(values, nbins, 0, view.maxCost, true)
                });
            }
        });

        // update ticks and tick lines
        var $densityPlot = $("#density-plot");
        var dticksGroup = d3.select("#density-plot > g:nth-of-type(1)");
        var maxBin = d3.max(series.map(function(d) { return d3.max(d.histogram.bins); }));
        var xScale = d3.scale.linear().domain([0, view.maxCost]).rangeRound([0, $densityPlot.innerWidth() - 14]);
        var yScale = d3.scale.linear().domain([0, maxBin]).rangeRound([18, $densityPlot.innerHeight()]);
        var xTicks = [];

        for(var i = 0; i < nbins + 1; i += 1) {
            xTicks[i] = i * (view.maxCost / nbins);
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
                    return d.selection.color();
                }
            })
            .each(function(d) {
                var dself = d3.select(this);

                $(d.selection).bind("highlight", function(e, state) {
                    dself.classed("highlighted", state);
                });
            })
            .selectAll("rect")
            .data(function(d) {
                return d.histogram.bins.map(function(value, i) {
                    return {value: value, range: d.histogram.ranges[i]};
                });
            })
            .enter()
            .append("svg:rect")
            .attr("x", function(d) { return xScale(d.range.left); })
            .attr("y", function(d) { return yScale.range()[1]; })
            .attr("width", function(d) { return xScale(d.range.right) - xScale(d.range.left); })
            .attr("height", 0)
            .on("mouseover", function(d) {
                var right = d3.select(this).attr("x") > $densityPlot.innerWidth() / 2;

                d3.select("#bar-label")
                    .attr("x", xScale(right ? d.range.right : d.range.left))
                    .attr("y", yScale.range()[0] + 18)
                    .attr("text-anchor", right ? "end" : "start")
                    .text("%.0f--%.0f".format(d.range.left, d.range.right));
            })
            .on("mouseout", function(d) {
                d3.select("#bar-label").text("");
            })
            .transition()
            .duration(500)
            .attr("y", function(d) {
                return yScale.range()[1] + yScale.range()[0] - yScale(d.value);
            })
            .attr("height", function(d) { return yScale(d.value); });
        dSeries
            .selectAll("rect")
            .data(function(d) {
                return d.histogram.bins.map(function(value, i) {
                    return {value: value, range: d.histogram.ranges[i]};
                });
            })
            .transition()
            .duration(500)
            .attr("y", function(d) {
                return yScale.range()[1] + yScale.range()[0] - yScale(d.value);
            })
            .attr("height", function(d) { return yScale(d.value); });
        dSeries
            .exit()
            .transition()
            .duration(500)
            .remove()
            .selectAll("rect")
            .attr("y", function(d) { return yScale.range()[1]; })
            .attr("height", 0);
    };

    view.onload(function() {
        // basic properties
        view.maxCost = d3.max(view.runs, function(d) {
            return d3.max(d.runs, function(e) { return e.cost; });
        });

        view.selections = [];
    });

    view.onload(function() {
        // compute instance node positions
        var nodes = [];
        var xDomain = [null, null];
        var yDomain = [null, null];

        for(var i = 0; i < view.instances.length; i += 1) {
            var node = {
                x: this.projection[i][0],
                y: this.projection[i][1],
                name: this.instances[i],
                dominant: 0
            };
            var belongs = this.membership[i];
            var clusters = belongs.length;

            for(var j = 0; j < belongs.length; j += 1) {
                if(belongs[j] > belongs[node.dominant]) {
                    node.dominant = j;
                }
            }

            if(xDomain[0] === null || node.x < xDomain[0]) { xDomain[0] = node.x; }
            if(xDomain[1] === null || node.x > xDomain[1]) { xDomain[1] = node.x; }
            if(yDomain[0] === null || node.y < yDomain[0]) { yDomain[0] = node.y; }
            if(yDomain[1] === null || node.y > yDomain[1]) { yDomain[1] = node.y; }

            nodes[i] = node;
        }

        // prepare display area
        var markup = [
            '<svg id="cluster-view">',
            '</svg>'
        ];
        var $clusterView = $(markup.join("\n")).appendTo("#display");
        var dclusterView = d3.select("#cluster-view");

        // generate node shapes
        var paths = [
            "M 5 5 V -5 H -5 V 5 Z", // square
            "M 0 -%1$s L %1$s 0 L 0 %1$s L -%1$s 0 Z".format(Math.sqrt(50)), // diamond
            "M 0 -7 L 0 7 M -7 0 L 7 0", // cross
            "M 5 -5 L -5 5 M -5 -5 L 5 5", // tilted cross
            "M 0 0 L 10 0 L 0 -10 Z", // bottom-left right triangle
            "M 0 0 L -10 0 L 0 -10 Z", // bottom-right right triangle
            "M 0 0 L -10 0 L 0 10 Z", // top-right right triangle
            "M 0 0 L 10 0 L 0 10 Z", // top-left right triangle
            "M -5 0 L 5 0 L 0 -%s Z".format(Math.sqrt(75)), // upward equilateral triangle
            "M -5 0 L 5 0 L 0 %s Z".format(Math.sqrt(75)), // downward equilateral triangle
            "M 0 -5 L 0 5 L %s 0 Z".format(Math.sqrt(75)), // rightward equilateral triangle
            "M 0 -5 L 0 5 L -%s 0 Z".format(Math.sqrt(75)), // leftward equilateral triangle
            "M -7 0 L 7 0 M 0 0 L 0 9", // downward tick
            "M -7 0 L 7 0 M 0 0 L 0 -9", // upward tick
            "M 0 -7 L 0 7 M 0 0 L 9 0", // rightward tick
            "M 0 -7 L 0 7 M 0 0 L -9 0" // leftward tick
        ];

        dclusterView
            .append("svg:defs")
            .selectAll("path")
            .data(paths)
            .enter()
            .append("svg:path")
            .attr("id", function(d, i) { return "node-shape-%s".format(i); })
            .attr("d", function(d) { return d; });

        // add instance nodes
        var xScale = d3.scale.linear().domain(xDomain).rangeRound([16, $clusterView.innerWidth() - 16]);
        var yScale = d3.scale.linear().domain(yDomain).rangeRound([16, $clusterView.innerHeight() - 16]);

        dclusterView
            .selectAll(".instance-point")
            .data(nodes)
            .enter()
            .append("svg:use")
            .attr("xlink:href", function(d) { return "#node-shape-%s".format(d.dominant % paths.length); })
            .attr("class", "instance-point")
            .attr("x", function(d) { return xScale(d.x) + 0.5; })
            .attr("y", function(d) { return yScale(d.y) + 0.5; })
            .attr("height", 10)
            .attr("width", 10)
            .on("mouseover", function(d) {
                var dself = d3.select(this);
                var right = dself.attr("x") > $clusterView.innerWidth() / 2;

                dself.classed("highlighted", true);

                dtip
                    .attr("x", xScale(d.x) + (right ? -12.5 : 12.5))
                    .attr("y", yScale(d.y) + 0.5)
                    .style("text-anchor", right ? "end" : "start")
                    .text(d.name);
            })
            .on("mouseout", function() {
                d3.select(this).classed("highlighted", false);

                dtip.text("");
            });

        // prepare tooltip
        var dglow =
            dclusterView
                .append("svg:defs")
                .append("svg:filter")
                .attr("id", "soft-glow")
                .attr("width", "150%")
                .attr("height", "150%");

        dglow
            .append("svg:feColorMatrix")
            .attr("in", "SourceGraphic")
            .attr("result", "matrixOut")
            .attr("type", "matrix")
            .attr("values", "0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0.5")
            .attr("stdDeviation", 2);
        dglow
            .append("svg:feGaussianBlur")
            .attr("in", "matrixOut")
            .attr("result", "blurOut")
            .attr("stdDeviation", 2);
        dglow
            .append("svg:feBlend")
            .attr("in", "SourceGraphic")
            .attr("in2", "blurOut")
            .attr("mode", "normal");

        var dtip =
            dclusterView
                .append("svg:text")
                .attr("filter", "url(#soft-glow)")
                .style("font-weight", "bold");
    });

    view.onload(function() {
        // build the configuration controls
        $('<p>Histogrammed:<select id="density-select"></select></p>').appendTo("#configuration-section > div");

        d3.select("#density-select")
            .selectAll("option")
            .data(this.solvers)
            .enter()
            .append("option")
            .attr("value", function(d) { return d; })
            .text(function(d) { return d; });

        $("#density-select")
            .selectmenu({ style: "dropdown" })
            .change(function(event) { $(view).trigger("selections-changed"); });
    });

    view.onload(function() {
        // mise en place
        var colors = d3.scale.category10();
        var count = 0;

        // prepare the selection list
        $("<p>Selections:</p>")
            .appendTo("#configuration-section > div")
            .append('<ul id="selections-list"></ul>');

        var updateSelectionList = function() {
            // populate list
            var items = 
                d3.select("#selections-list")
                    .selectAll("li")
                    .data(view.selections, function(d) { return d.number; });

            var added =
                items
                    .enter()
                    .append("li")
                    .each(function(d) {
                        var $d = $(d);

                        $(this)
                            .mouseenter(function() { $d.trigger("highlight", [true]); })
                            .mouseleave(function() { $d.trigger("highlight", [false]); });
                    });

            items.exit().remove();

            // set up UI
            added
                .append("span")
                .attr("title", "Toggle")
                .attr("class", "ui-icon ui-icon-pause")
                .on("click", function(d) {
                    // switch the icon and text
                    $(this).toggleClass("ui-icon-pause");
                    $(this).toggleClass("ui-icon-play");
                    $(this).parent().toggleClass("hidden");

                    // hide the region
                    d.toggleVisible();
                });

            added
                .append("span")
                .filter(function(d) { return d.number > 0; })
                .attr("title", "Remove")
                .attr("class", "ui-icon ui-icon-close")
                .on("click", function(d) { d.remove(); });

            added
                .append("span")
                .html(function(d) { return d.titleMarkup; })
                .style("color", function(d) { return d.color(); });
        };

        $(view).bind("selections-changed", function() { updateSelectionList(); });

        // handle selection-region interaction
        var makeSelection = function(point) {
            var newt = {
                area: function() {
                    return (this.p2.x - this.p1.x) * (this.p2.y - this.p1.y);
                },
                color: function() {
                    return colors(this.number % 10);
                },
                update: function(q) {
                    this.p1 = {x: Math.min(this.p0.x, q.x), y: Math.min(this.p0.y, q.y)};
                    this.p2 = {x: Math.max(this.p0.x, q.x), y: Math.max(this.p0.y, q.y)};

                    region
                        .attr("x", this.p1.x)
                        .attr("y", this.p1.y)
                        .attr("height", this.p2.y - this.p1.y)
                        .attr("width", this.p2.x - this.p1.x);
                },
                add: function() {
                    view.selections.push(selection);

                    $(view).trigger("selections-changed");
                },
                remove: function() {
                    region.remove();

                    var index = view.selections.indexOf(this);

                    if(index >= 0) {
                        view.selections.splice(index, 1);

                        $(view).trigger("selections-changed");
                    }
                },
                show: function() {
                    this.visible = true;

                    if(region !== undefined) {
                        $(region.node()).show();
                    }

                    $(view).trigger("selections-changed");
                },
                hide: function() {
                    this.visible = false;

                    if(region !== undefined) {
                        $(region.node()).hide();
                    }

                    $(view).trigger("selections-changed");
                },
                toggleVisible: function() {
                    if(this.visible) {
                        this.hide();
                    }
                    else {
                        this.show();
                    }
                },
                instances: function() {
                    var selected = d3.selectAll("#cluster-view .instance-point");

                    if(this.p0 !== undefined) {
                        var self = this;

                        var inside = function(point) {
                            var cx = this.x.animVal.value;
                            var cy = this.y.animVal.value;

                            return true &&
                                cx >= self.p1.x && cx <= self.p2.x &&
                                cy >= self.p1.y && cy <= self.p2.y;
                        };

                        selected = selected.filter(inside);
                    }

                    return selected[0].map(function(d) { return d.__data__.name; });
                },
                p0: point,
                number: count,
                visible: true,
                titleMarkup: 'Region <span class="region-number">#' + count + '</span>'
            };

            if(point !== undefined) {
                var region =
                    d3.select("#cluster-view")
                        .append("svg:rect")
                        .attr("class", "selection-region")
                        .style("fill", newt.color())
                        .style("stroke", newt.color())
                        .on("mouseover", function() { $(newt).trigger("highlight", [true]); })
                        .on("mouseout", function() { $(newt).trigger("highlight", [false]); });

                newt.update(point);

                $(newt).bind("highlight", function(e, state) {
                    region.classed("highlighted", state);
                });
            }

            count += 1;

            return newt;
        };

        // the implicit "everything" selection
        var totalSelection = makeSelection();

        totalSelection.titleMarkup = "All Instances";

        view.selections = [totalSelection];

        // the current selection, if any
        var selection = null;

        var finish = function() {
            if(selection.area() >= 25) {
                selection.add();
            }
            else {
                selection.remove();
            }

            selection = null;
        }

        $("#cluster-view")
            .mousedown(function(event) {
                if(selection === null) {
                    selection = makeSelection({x: event.layerX, y: event.layerY});
                }
                else {
                    finish();
                }
            })
            .mousemove(function(event) {
                if(selection !== null) {
                    selection.update({x: event.layerX, y: event.layerY});
                }
            })
            .mouseup(function(event) {
                if(selection !== null) {
                    finish();
                }
            });
    });

    view.onload(function() {
        // prepare for plotting
        var dPlot =
            d3.select("body")
                .append("div")
                .attr("id", "density-plot-area")
                .append("svg:svg")
                .attr("id", "density-plot");

        dPlot.append("svg:g");
        dPlot.append("svg:g");

        var dlabelsGroup = dPlot.append("svg:g");

        dlabelsGroup
            .append("svg:text")
            .attr("x", 4)
            .attr("y", 2)
            .attr("dy", "1em")
            .text("Time to Solution (CPU s) -->");

        dlabelsGroup
            .append("svg:g")
            .attr(
                "transform",
                "translate(%s, %s) rotate(90)".format(
                    $("#density-plot").innerWidth() - 10,
                    20
                )
            )
            .append("svg:text")
            .text("Fraction of Runs -->");

        dlabelsGroup
            .append("svg:text")
            .attr("id", "bar-label")
            .attr("filter", "url(#soft-glow)")
            .style("font-weight", "bold")
            .text("");

        $(view).bind("selections-changed", function() { view.updatePlot(); });
    });

    view.onload(function() {
        // add projection area outline
        var dclusterView = d3.select("#cluster-view");
        var $clusterView = $(dclusterView.node());

        dclusterView
            .append("svg:rect")
            .attr("id", "cluster-view-border")
            .attr("x", 1)
            .attr("y", 1)
            .attr("height", $clusterView.innerHeight() - 2)
            .attr("width", $clusterView.innerWidth() - 2)
            .style("stroke", view.selections[0].color());

        // draw everything for the first time
        $(view).trigger("selections-changed");
    });

    view.unload = function() {
        $("#display").empty();
        $("#density-plot-area").remove();
        $("#configuration-section > div").empty();
    };

    return view;
};

