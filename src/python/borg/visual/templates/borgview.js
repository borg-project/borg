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

            console.log("loaded " + request.path + " (" + completed + " of " + loadable.resources.length + ")");

            if (completed == loadable.resources.length) {
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

var newTableView = function(ui, category) {
    var view = {
        sort: null,
        top_instance_index: null,
        resources: [
            {name: "runs", path: "data/" + category.path + "/runs.json"},
            {name: "solvers", path: "data/" + category.path + "/solvers.json"},
            {name: "instances", path: "data/" + category.path + "/instances.json"},
            {name: "similarity", path: "data/" + category.path + "/similarity.json"}
        ]
    };
    var sorts = {
        byName: function (a, b) {
            return d3.ascending(a.instance, b.instance);
        },
        byTotalCost: function (a, b) {
            return d3.descending(a.total_cost, b.total_cost);
        },
        bySimilarity: function (a, b) {
            var ai = renderer.instances.indexOf(a.instance);
            var bi = renderer.instances.indexOf(b.instance);
            var ti = renderer.top_instance_index;

            return d3.descending(this.similarity[ti][ai], this.similarity[ti][bi]);
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

    view.loaded = function () {
        // mise en place
        var $runsTable = $("#runs-table");
        var $tableHeader = $("#runs-table > header");

        $runsTable.show();

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
                            .css({
                                position: "fixed",
                                top: headerTop
                            })
                            .width(tableWidth - 4) // XXX hard-coded constant
                            .data("overlaid", true);
                    }
                }
                else {
                    if($tableHeader.data("overlaid")) {
                        $("#runs-table").css("margin-top", "");

                        $tableHeader
                            .css({
                                position: "static",
                                top: ""
                            })
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
                this.reorder(sorts[$(this).val()]);
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

                this.reorder(this.sort);
            });

        // handle row-click events
        $("body")
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
        $("#runs-table").hide();

        d3.selectAll("#runs-list > li").remove();
        d3.selectAll("#runs-table > header > .solver-name-outer").remove();
    };

    return view;
};

//
// PROJECTION VIEW
//

var newProjectionView = function (ui) {
    var view = {
        resources: [
            {name: "runs", path: "data/" + ui.category.path + "/runs.json"},
            {name: "solvers", path: "data/" + ui.category.path + "/solvers.json"},
            {name: "instances", path: "data/" + ui.category.path + "/instances.json"},
            {name: "projection", path: "data/" + ui.category.path + "/projection.json"}
        ]
    };

    view.loaded = function () {
        var $projection = $("#instance-projection");

        var xScale = d3.scale.linear().domain([-1, 1]).range([0, $projection.innerWidth()]);
        var yScale = d3.scale.linear().domain([-1, 1]).range([0, $projection.innerHeight()]);

        var runsRows = 
            d3.select("#instance-projection")
                .selectAll("circle")
                .data(this.projection)
                .enter()
                .append("svg:circle")
                .attr("cx", function(d) { return xScale(d[0]); })
                .attr("cy", function(d) { return yScale(d[1]); })
                .attr("r", 8);

        $projection.show();
    };

    view.unload = function() {
        d3.selectAll("#instance-projection *").remove();

        $("#instance-projection").hide();
    };

    return view;
};

//
// INTERFACE GLUE
//

var viewFactories = [
    {name: "table", text: "Table View", build: newTableView},
    {name: "projection", text: "Projection View", build: newProjectionView},
];
var ui = {
    rootURL: "{{ base_url }}",
    view: null,
    viewFactory: viewFactories[1],
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
    this.view = factory.build(this, this.category);

    load(this.view, callback);
};

load(ui);

});

