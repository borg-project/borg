$(function () {

//
// GENERAL SUPPORT
//

var load = function (loadable, callback) {
    var completed = 0;
    var makeHandler = function (request) {
        return function (fetched) {
            loadable[request.name] = fetched;

            completed += 1;

            if (completed == loadable.resources.length) {
                console.log("finished loading " + completed + " resource(s)");

                if (callback !== undefined) {
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

    view.loaded = function () {
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
        $(function () {
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
            .text(function (name) { return name; });

        var nameWidths = $("div.solver-name-outer").map(function (i, o) { return $(o).outerWidth(); });

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
            .data(function (datum) { return [datum.instance]; })
            .enter()
            .append("div")
            .attr("class", "instance-title")
            .text(function (title) { return title; });

        runsRows
            .selectAll(".instance-run")
            .data(function (datum) { return datum.runs; })
            .enter()
            .append("div")
            .attr("class", "instance-run")
            .attr("title", function (datum) { return datum.cost; })
            .style(
                "background-color",
                function (datum) {
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
            .change(function (event, ui) {
                view.reorder(sorts[$(this).val()]);
            });

        // set up row interface UI
        $("button.raise-button")
            .button({
                icons: { primary: "ui-icon-arrowthickstop-1-n" },
                text: false
            })
            .click(function () {
                $("li.ui-selected")
                    .detach()
                    .prependTo("#runs-list");

                view.reorder(view.sort);
            });

        // handle row-click events
        $("body") // XXX correctly remove this handler on unload
            .click(function () {
                $("li.ui-selected").removeClass("ui-selected");
                $("#row-interface").hide();
            });
        $("li.instance-runs")
            .click(function () {
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

    view.unload = function () {
        $("#display").empty();
    };

    var sorts = {
        byName: function (a, b) {
            return d3.ascending(a.instance, b.instance);
        },
        byTotalCost: function (a, b) {
            return d3.descending(a.total_cost, b.total_cost);
        },
        bySimilarity: function (a, b) {
            var ai = view.instances.indexOf(a.instance);
            var bi = view.instances.indexOf(b.instance);
            var ti = view.top_instance_index;

            return d3.descending(view.similarity[ti][ai], view.similarity[ti][bi]);
        }
    };

    view.reorder = function (sort) {
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

var newClusterView = function (ui) {
    var view = {
        resources: [
            {name: "runs", path: "data/" + ui.category.path + "/runs.json"},
            {name: "solvers", path: "data/" + ui.category.path + "/solvers.json"},
            {name: "instances", path: "data/" + ui.category.path + "/instances.json"},
            {name: "membership", path: "data/" + ui.category.path + "/membership.json"}
        ]
    };

    view.loaded = function () {
        // compute membership vectors
        // XXX don't hard-code cluster count and increment
        var vectors = [];
        var clusters = 16;
        var clustersPerSide = clusters / 4;
        var increment = 0.25;

        for (var i = 0; i < clustersPerSide; i += 1) {
            vectors[i] = {x: i * increment - 0.5, y: -0.5};
            vectors[i + clustersPerSide] = {x: 0.5, y: i * increment - 0.5};
            vectors[i + clustersPerSide * 2] = {x: 0.5 - i * increment, y: 0.5};
            vectors[i + clustersPerSide * 3] = {x: -0.5, y: i * increment - 0.5};
        }

        // compute instance node positions
        var nodes = [];
        var xDomain = [null, null];
        var yDomain = [null, null];

        for (var i = 0; i < view.instances.length; i += 1) {
            var node = {x: 0.5, y: 0.5, name: view.instances[i], dominant: 0};

            for (var j = 0; j < clusters; j += 1) {
                var m = view.membership[i][j];

                if (m > view.membership[i][node.dominant]) {
                    node.dominant = j;
                }

                node.x += m * vectors[j].x;
                node.y += m * vectors[j].y;
            }

            if (xDomain[0] === null || node.x < xDomain[0]) {
                xDomain[0] = node.x;
            }
            if (xDomain[1] === null || node.x > xDomain[1]) {
                xDomain[1] = node.x;
            }
            if (yDomain[0] === null || node.y < yDomain[0]) {
                yDomain[0] = node.y;
            }
            if (yDomain[1] === null || node.y > yDomain[1]) {
                yDomain[1] = node.y;
            }

            nodes[i] = node;
        }

        // position nodes on screen
        var $clusterView = $('<svg id="cluster-view"></svg>').appendTo("#display");
        var xScale = d3.scale.linear().domain(xDomain).range([8, $clusterView.innerWidth() - 8]);
        var yScale = d3.scale.linear().domain(yDomain).range([8, $clusterView.innerHeight() - 8]);
        var colors = d3.scale.category20();

        d3.select("#cluster-view")
            .selectAll("circle")
            .data(nodes)
            .enter()
            .append("svg:circle")
            .attr("r", 8)
            .attr("cx", function (d) { return xScale(d.x); })
            .attr("cy", function (d) { return yScale(d.y); })
            .style("fill", function (d) { return colors(d.dominant / 16.0); })
            .style("fill-opacity", 0.5);

        // build the configuration controls
        var $densitySelect = $('<select id="density-select"></select>');

        $densitySelect.appendTo("#configuration-section > div");

        d3.select("#density-select")
            .selectAll("option")
            .data(this.solvers)
            .enter()
            .append("option")
            .attr("value", function (d) { return d; })
            .text(function (d) { return d; });

        $densitySelect
            .selectmenu({ style: "dropdown" })
            .change(function (event) {
                view.updatePlot($(this).val());
            });

        // build the density plot
        this.updatePlot(this.solvers[0], true);

        // prepare for selection areas
        var selectionPoints = [];
        var selectionString = function () {
            return "" +
                selectionPoints[0].x + "," + selectionPoints[0].y + " " +
                selectionPoints[1].x + "," + selectionPoints[0].y + " " +
                selectionPoints[1].x + "," + selectionPoints[1].y + " " +
                selectionPoints[0].x + "," + selectionPoints[1].y;
        };

        $clusterView
            .mousedown(function (event) {
                selectionPoints[0] = {x: event.layerX, y: event.layerY};
                selectionPoints[1] = selectionPoints[0];
                console.log(selectionString());

                d3.select("#cluster-view")
                    .selectAll("polygon")
                    .data([0])
                    .enter()
                    .append("svg:polygon")
                    .attr("id", "in-progress-selection")
                    .attr("points", selectionString());
            })
            .mousemove(function (event) {
                // only do this if a selection is extant
                var selection = d3.select("#in-progress-selection");

                if (!selection.empty()) {
                    selectionPoints[1] = {x: event.layerX, y: event.layerY};

                    selection.attr("points", selectionString());
                }
            })
            .mouseup(function (event) {
                d3.select("#in-progress-selection").remove();
            });
    };

    view.updatePlot = function (solver, initialize) {
        // build an empty bin vector
        var bins = [];
        var binCount = 40;

        for (var i = 0; i < binCount; i += 1) {
            bins[i] = 0;
        }

        // prepare the DOM, if need be
        if (initialize) {
            $("<div id=\"density-plot-area\"></div>")
                .appendTo("body")
                .append("<svg id=\"density-plot\"></svg>");

            d3.select("#density-plot")
                .selectAll("rect")
                .data(bins)
                .enter()
                .append("svg:rect");
        }

        // populate the bins
        var getMaxCost = function (d) { return d3.max(d.runs, function (c) { return c.cost; }); };
        var maxCost = d3.max(this.runs, getMaxCost);
        var binSize = maxCost / binCount;

        for (var i = 0; i < this.runs.length; i += 1) {
            var instanceRuns = this.runs[i].runs;

            for (var j = 0; j < instanceRuns.length; j += 1) {
                var run = instanceRuns[j];

                if (run.solver == solver) {
                    var bin = Math.floor(run.cost / binSize);

                    bins[bin] += 1;
                }
            }
        }

        var maxBin = d3.max(bins);
        var barWidth = $("#density-plot").innerWidth() / binCount;
        var barMaxHeight = $("#density-plot").innerHeight();

        // draw the bars
        d3.selectAll("#density-plot rect")
            .data(bins)
            .attr("x", function (d, i) { return i * barWidth; })
            .attr("y", function (d) { return barMaxHeight - (d / maxBin) * barMaxHeight; })
            .attr("width", barWidth)
            .attr("height", function (d) { return (d / maxBin) * barMaxHeight; });
    };

    view.unload = function () {
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

ui.loaded = function () {
    // enable change-category links
    d3.select("#change-data > nav")
        .selectAll("a")
        .data(this.categories)
        .enter()
        .append("a")
        .attr("href", "")
        .text(function(d) { return d.name; });

    $("#change-data > nav > a")
        .click(function (event) {
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
        .click(function (event) {
            event.preventDefault();

            ui.changeView(this.__data__);
        });

    // get the ball rolling
    this.changeCategory(this.categories[0]);
};

ui.changeCategory = function (category) {
    this.category = category;

    this.changeView(this.viewFactory);
};

ui.changeView = function (factory) {
    console.log("changing to category " + this.category.name + ", " + factory.name + " view");

    // update our location
    window.history.pushState(null, null, "ui/" + this.category.path + "/" + factory.name);

    // prepare to unload old view
    var callback = undefined;

    if (this.view !== null) {
        var old = this.view;

        callback = function () { old.unload(); };
    }

    // switch to the new view
    this.viewFactory = factory;
    this.view = factory.build(this);

    load(this.view, callback);
};

load(ui);

});

