"use strict";

$(function() {

//
// GENERAL SUPPORT
//

var load = function(loadable, callback) {
    var completed = 0;
    var makeHandler = function(request) {
        return function(fetched) {
            loadable[request.name] = fetched;

            completed += 1;

            if(completed == loadable.resources.length) {
                console.log("finished loading " + completed + " resource(s)");

                if(callback !== undefined) {
                    callback();
                }

                loadable.loaded();
            }
        };
    };

    for(var i = 0; i < loadable.resources.length; i += 1) {
        var request = loadable.resources[i];

        d3.json(request.path, makeHandler(request));
    }
};

//
// VIEW BASE
//

var newView = function(resources) {
    return {
        onload: function(handler) {
            this.loadHandlers.push(handler);
        },
        loaded: function() {
            for(var i = 0; i < this.loadHandlers.length; i += 1) {
                this.loadHandlers[i].call(this);
            }
        },
        resources: resources,
        loadHandlers: []
    };
};

//
// TABLE VIEW
//

var newTableView = function(ui) {
    var view = {
        sort: null,
        top_instance_index: null,
        resources: [
            {name: "runs", path: "data/" + ui.category.path + "/runs.json"},
            {name: "solvers", path: "data/" + ui.category.path + "/solvers.json"},
            {name: "instances", path: "data/" + ui.category.path + "/instances.json"},
            {name: "similarity", path: "data/" + ui.category.path + "/similarity.json"}
        ]
    };

    view.loaded = function() {
        // mise en place
        var markup = [
            '<section id="runs-table">',
            '    <header>',
            '        <div id="header-ui-controls">',
            '            <!--<label for="row-order-select">Order Rows By:</label>-->',
            '            <select id="row-order-select">',
            '                <option value="byName">Sort by Name</option>',
            '                <option value="byTotalCost">Sort by Total Cost</option>',
            '                <option value="bySimilarity">Sort by Similarity</option>',
            '            </select>',
            '        </div>',
            '    </header>',
            '    <ul id="runs-list"></ul>',
            '    <div id="row-interface">',
            '        <button class="raise-button">Raise</button>',
            '    </div>',
            '</section>'
        ];
        var $runsTable = $(markup.join("\n")).appendTo("#display");
        var $tableHeader = $("#runs-table > header");

        // compute run costs
        for(var i = 0; i < this.runs.length; i += 1) {
            var run = this.runs[i];

            run.total_cost = 0.0;

            for(var j = 0; j < run.runs.length; j += 1) {
                run.total_cost += run.runs[j].cost;
            }
        }

        // keep the table header in view
        $(function() {
            var headerTop = $tableHeader.offset().top;

            $(window).scroll(function() {
                if($(document).scrollTop() > headerTop) {
                    if(!$tableHeader.data("overlaid")) {
                        var tableWidth = $("#runs-table").outerWidth();

                        $("#runs-table").css("margin-top", $tableHeader.outerHeight());

                        $tableHeader
                            .css({position: "fixed", top: headerTop})
                            .width(tableWidth - 4) // XXX hard-coded constant
                            .data("overlaid", true);
                    }
                }
                else {
                    if($tableHeader.data("overlaid")) {
                        $("#runs-table").css("margin-top", "");

                        $tableHeader
                            .css({position: "static", top: ""})
                            .data("overlaid", false);
                    }
                }
            });
        });

        // render column headings
        d3.select("#runs-table > header")
            .selectAll(".solver-name-outer")
            .data(this.solvers)
            .enter()
            .append("div")
            .attr("class", "solver-name-outer")
            .append("div")
            .attr("class", "solver-name")
            .text(function(name) { return name; });

        var nameWidths = $("div.solver-name-outer").map(function(i, o) { return $(o).outerWidth(); });

        $("#runs-table > header").height(d3.max(nameWidths));
        $("div.solver-name-outer").width("25px");
        $("div.solver-name").css({
            "-webkit-transform": "rotate(90deg)",
            "-moz-transform": "rotate(90deg)",
            "-ms-transform": "rotate(90deg)",
            "-o-transform": "rotate(90deg)",
            "transform": "rotate(90deg)"
        });

        // render runs table
        var runsRows = 
            d3.select("#runs-list")
                .selectAll(".instance-runs")
                .data(this.runs)
                .enter()
                .append("li")
                .attr("class", "instance-runs");

        var maxCost = 6100.0; // XXX
        var costScale = d3.scale.linear().domain([0.0, maxCost]);
        var costColor = d3.interpolateRgb("rgb(237, 248, 177)", "rgb(44, 127, 184)");

        runsRows
            .selectAll(".instance-title")
            .data(function(datum) { return [datum.instance]; })
            .enter()
            .append("div")
            .attr("class", "instance-title")
            .text(function(title) { return title; });

        runsRows
            .selectAll(".instance-run")
            .data(function(datum) { return datum.runs; })
            .enter()
            .append("div")
            .attr("class", "instance-run")
            .attr("title", function(datum) { return datum.cost; })
            .style(
                "background-color",
                function(datum) {
                    if(datum.answer != null)
                        return costColor(costScale(datum.cost));
                    else
                        return "black";
                }
            );

        // by default, sort by instance name
        this.reorder(sorts.byName);

        // set up overall UI
        $("#row-order-select")
            .selectmenu({ style: "dropdown" })
            .change(function(event, ui) {
                view.reorder(sorts[$(this).val()]);
            });

        // set up row interface UI
        $("button.raise-button")
            .button({
                icons: { primary: "ui-icon-arrowthickstop-1-n" },
                text: false
            })
            .click(function() {
                $("li.ui-selected")
                    .detach()
                    .prependTo("#runs-list");

                view.reorder(view.sort);
            });

        // handle row-click events
        $("body") // XXX correctly remove this handler on unload
            .click(function() {
                $("li.ui-selected").removeClass("ui-selected");
                $("#row-interface").hide();
            });
        $("li.instance-runs")
            .click(function() {
                $(this)
                    .addClass("ui-selected")
                    .siblings()
                    .removeClass("ui-selected");

                $("#row-interface")
                    .css({ top: $(this).position().top + $(this).outerHeight() })
                    .show();

                event.stopPropagation();
            });
    };

    view.unload = function() {
        $("#display").empty();
    };

    var sorts = {
        byName: function(a, b) {
            return d3.ascending(a.instance, b.instance);
        },
        byTotalCost: function(a, b) {
            return d3.descending(a.total_cost, b.total_cost);
        },
        bySimilarity: function(a, b) {
            var ai = view.instances.indexOf(a.instance);
            var bi = view.instances.indexOf(b.instance);
            var ti = view.top_instance_index;

            return d3.descending(view.similarity[ti][ai], view.similarity[ti][bi]);
        }
    };

    view.reorder = function(sort) {
        console.log("reordering by " + sort);

        this.sort = sort;

        var runsRows = d3.selectAll("#runs-list .instance-runs");

        this.top_instance_index = this.instances.indexOf(runsRows[0][0].__data__.instance);

        runsRows.sort(sort);

        this.top_instance_index = this.instances.indexOf(runsRows[0][0].__data__.instance);
    };

    return view;
};

//
// CLUSTER VIEW
//

var newClusterView = function(ui) {
    var view = newView([
        ui.resource("runs", "runs.json"),
        ui.resource("solvers", "solvers.json"),
        ui.resource("instances", "instances.json"),
        ui.resource("membership", "membership.json"),
        ui.resource("projection", "projection.json"),
    ]);

    view.selections = [];

    view.onload(function() {
        // compute membership vectors
        // XXX don't hard-code cluster count
        //var vectors = [];
        var clusters = 15;
        //var angle = 2 * Math.PI / clusters;

        //for(var i = 0; i < clusters; i += 1) {
            //vectors[i] = {x: Math.sin(angle * i) / 2, y: Math.cos(angle * i) / 2};
        //}

        // compute instance node positions
        var nodes = [];
        var xDomain = [null, null];
        var yDomain = [null, null];

        for(var i = 0; i < view.instances.length; i += 1) {
            //var node = {x: 0.5, y: 0.5, name: view.instances[i], dominant: 0};

            //for(var j = 0; j < clusters; j += 1) {
                //var m = view.membership[i][j];

                //if(m > view.membership[i][node.dominant]) {
                    //node.dominant = j;
                //}

                //node.x += m * vectors[j].x;
                //node.y += m * vectors[j].y;
            //}

            var node = {
                x: this.projection[i][0],
                y: this.projection[i][1],
                name: this.instances[i],
                dominant: 0
            };

            for(var j = 0; j < clusters; j += 1) {
                if(this.membership[i][j] > this.membership[i][node.dominant]) {
                    node.dominant = j;
                }
            }

            if(xDomain[0] === null || node.x < xDomain[0]) { xDomain[0] = node.x; }
            if(xDomain[1] === null || node.x > xDomain[1]) { xDomain[1] = node.x; }
            if(yDomain[0] === null || node.y < yDomain[0]) { yDomain[0] = node.y; }
            if(yDomain[1] === null || node.y > yDomain[1]) { yDomain[1] = node.y; }

            nodes[i] = node;
        }

        // prepare node shapes
        var $clusterView = $('<svg id="cluster-view"></svg>').appendTo("#display");
        // XXX
        //var defs = d3.select("#cluster-view").append("svg:defs");
        //var defs = d3.select("#cluster-view");

        //defs.append("svg:circle").attr("id", "foo-rect");

        // add cluster vectors
        var xScale = d3.scale.linear().domain(xDomain).range([8, $clusterView.innerWidth() - 8]);
        var yScale = d3.scale.linear().domain(yDomain).range([8, $clusterView.innerHeight() - 8]);
        var colors = d3.scale.category20();

        //d3.select("#cluster-view")
            //.selectAll(".cluster-arrow")
            //.data(vectors)
            //.enter()
            //.append("svg:line")
            //.attr("class", "cluster-arrow")
            //.attr("x1", function(d) { return xScale(0.5); })
            //.attr("y1", function(d) { return yScale(0.5); })
            //.attr("x2", function(d) { return xScale(0.5 + d.x); })
            //.attr("y2", function(d) { return yScale(0.5 + d.y); })
            //.style("stroke", function(d, i) { return colors(i / 16.0); });

        // add instance nodes
        d3.select("#cluster-view")
            .selectAll(".instance-point")
            .data(nodes)
            .enter()
            .append("svg:circle")
            //.attr("xlink:href", "#foo-rect")
            .attr("class", "instance-point")
            .attr("r", 8)
            .attr("cx", function(d) { return xScale(d.x); })
            .attr("cy", function(d) { return yScale(d.y); })
            .style("fill", function(d) { return colors(d.dominant / clusters); });
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
            .change(function(event) { view.drawHistogram(); });
    });

    view.onload(function() {
        // mise en place
        var colors = d3.scale.category10();
        var count = 0;

        // prepare the selection list
        $("<p>Selections:</p>")
            .appendTo("#configuration-section > div")
            .append('<ul id="selections-list"></ul>');

        var drawSelectionList = function() {
            // populate list
            var items = 
                d3.select("#selections-list")
                    .selectAll("li")
                    .data(view.selections, function(d) { return d.number; });

            var added =
                items
                    .enter()
                    .append("li")
                    .on("mouseover", function(d) { d.toggleHighlighted(); });

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

                    drawSelectionList();

                    view.drawHistogram();
                },
                remove: function() {
                    region.remove();

                    var index = view.selections.indexOf(this);

                    if(index >= 0) {
                        view.selections.splice(index, 1);

                        drawSelectionList();

                        view.drawHistogram();
                    }
                },
                show: function() {
                    this.visible = true;

                    if(region !== undefined) {
                        $(region.node()).show();
                    }

                    view.drawHistogram();
                },
                hide: function() {
                    this.visible = false;

                    if(region !== undefined) {
                        $(region.node()).hide();
                    }

                    view.drawHistogram();
                },
                toggleVisible: function() {
                    if(this.visible) {
                        this.hide();
                    }
                    else {
                        this.show();
                    }
                },
                toggleHighlighted: function() {
                    // XXX
                    //if(region !== undefined) {
                        //console.log("toggling");
                        //$(region.node()).toggleClass("highlighted");
                    //}
                },
                instances: function() {
                    var selected = d3.selectAll("#cluster-view .instance-point");

                    if(this.p0 !== undefined) {
                        var self = this;

                        var inside = function(point) {
                            var cx = this.cx.animVal.value;
                            var cy = this.cy.animVal.value;

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
                        .style("stroke", newt.color());

                newt.update(point);
            }

            count += 1;

            return newt;
        };

        // the implicit "everything" selection
        var totalSelection = makeSelection();

        totalSelection.titleMarkup = "All Instances";

        view.selections = [totalSelection];

        drawSelectionList();

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
        // daw the plot
        this.drawHistogram();
    });

    view.makeHistogram = function(solver, binCount, instances) {
        // initialize a histogram
        var histogram = {bins: []};

        for(var i = 0; i < binCount; i += 1) {
            histogram.bins.push(0);
        }

        // populate the bins
        var subset = this.runs.filter(function(d) { return instances.indexOf(d.instance) >= 0; });

        if(subset.length > 0) {
            var maxCost =
                d3.max(
                    subset,
                    function(d) {
                        return d3.max(d.runs, function(c) { return c.cost; });
                    }
                );
            var binSize = maxCost / binCount;

            for(var i = 0; i < subset.length; i += 1) {
                var instanceRuns = subset[i].runs;

                for(var j = 0; j < instanceRuns.length; j += 1) {
                    var run = instanceRuns[j];

                    if(run.solver == solver) {
                        histogram.bins[Math.floor(run.cost / binSize)] += 1;
                    }
                }
            }
        }

        // normalize counts to densities
        var maxBin = histogram.bins.reduce(function(a, b) { return a + b; });

        for(var i = 0; i < histogram.bins.length; i += 1) {
            histogram.bins[i] /= maxBin;
        }

        // ...
        return histogram;
    };

    view.drawHistogram = function() {
        // prepare the DOM
        var $densityPlot = $("#density-plot").empty();

        if($densityPlot.length == 0) {
            $("<div id=\"density-plot-area\"></div>")
                .appendTo("body")
                .append("<svg id=\"density-plot\"></svg>");

            $densityPlot = $("#density-plot");
        }

        // build the histograms
        var binCount = 40;
        var solver = $("#density-select").val();
        var series = [];

        for(var i = 0; i < this.selections.length; i += 1) {
            var selection = this.selections[i];

            if(selection.visible) {
                var instances = this.selections[i].instances();
                var histogram = this.makeHistogram(solver, binCount, instances);

                series.push({selection: selection, histogram: histogram});
            }
        }

        // draw the histograms
        var barWidth = Math.round($densityPlot.innerWidth() / binCount);
        var barMaxHeight = $densityPlot.innerHeight();

        d3.select("#density-plot")
            .selectAll("g")
            .data(series)
            .enter()
            .append("svg:g")
            .style("fill", function(d) {
                if(d.selection === null) {
                    return "#aaaaaa";
                }
                else {
                    return d.selection.color();
                }
            })
            .style("fill-opacity", 0.5)
            .selectAll("rect")
            .data(function(d) { return d.histogram.bins; })
            .enter()
            .append("svg:rect")
            .attr("class", "histogram-bar")
            .attr("x", function(d, i) { return i * barWidth + 0.5; })
            .attr("y", function(d) { return barMaxHeight - d * barMaxHeight; })
            .attr("width", barWidth)
            .attr("height", function(d) { return d * barMaxHeight; });
    };

    view.unload = function() {
        $("#display").empty();
        $("#density-plot-area").remove();
        $("#configuration-section > div").empty();
    };

    return view;
};

//
// INTERFACE GLUE
//

var viewFactories = [
    {name: "cluster", text: "Cluster View", build: newClusterView},
    {name: "table", text: "Table View", build: newTableView},
];
var ui = {
    view: null,
    viewFactory: viewFactories[0],
    category: null,
    resources: [{name: "categories", path: "categories.json"}]
};

ui.resource = function(name, filename) {
    return {name: name, path: "data/" + this.category.path + "/" + filename};
};

ui.loaded = function() {
    // enable change-category links
    d3.select("#change-data > nav")
        .selectAll("a")
        .data(this.categories)
        .enter()
        .append("a")
        .attr("href", "")
        .text(function(d) { return d.name; });

    $("#change-data > nav > a")
        .click(function(event) {
            event.preventDefault();

            ui.changeCategory(this.__data__);
        });

    // enable change-view links
    d3.select("#change-view > nav")
        .selectAll("a")
        .data(viewFactories)
        .enter()
        .append("a")
        .attr("href", "")
        .text(function(d) { return d.text; });

    $("#change-view > nav > a")
        .click(function(event) {
            event.preventDefault();

            ui.changeView(this.__data__);
        });

    // get the ball rolling
    this.changeCategory(this.categories[0]);
};

ui.changeCategory = function(category) {
    this.category = category;

    this.changeView(this.viewFactory);
};

ui.changeView = function(factory) {
    console.log("changing to category " + this.category.name + ", " + factory.name + " view");

    // update our location
    window.history.pushState(null, null, "ui/" + this.category.path + "/" + factory.name);

    // prepare to unload old view
    var callback = undefined;

    if(this.view !== null) {
        var old = this.view;

        callback = function() { old.unload(); };
    }

    // switch to the new view
    this.viewFactory = factory;
    this.view = factory.build(this);

    load(this.view, callback);
};

load(ui);

});

