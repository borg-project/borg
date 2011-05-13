//// keep the table header in view
//$(function() {
    //var $tableHeader = $("#runs-table > header");
    //var headerTop = $tableHeader.offset().top;

    //$(window).scroll(function() {
        //if($(document).scrollTop() > headerTop) {
            //if(!$tableHeader.data("overlaid")) {
                //var tableWidth = $("#runs-table").outerWidth();

                //$("#runs-table").css("margin-top", $tableHeader.outerHeight());

                //$tableHeader
                    //.css({
                        //position: "fixed",
                        //top: headerTop
                    //})
                    //.width(tableWidth - 4) // XXX hard-coded constant
                    //.data("overlaid", true);
            //}
        //}
        //else {
            //if($tableHeader.data("overlaid")) {
                //$("#runs-table").css("margin-top", "");

                //$tableHeader
                    //.css({
                        //position: "static",
                        //top: ""
                    //})
                    //.data("overlaid", false);
            //}
        //}
    //});
//});

// define the table-oriented view
// XXX use a local self variable, obviously
borgView = {
    loaded: 0,
    preprocess: function() {
        // compute run costs
        for(var i = 0; i < borgView.runs.length; i += 1) {
            var run = borgView.runs[i];

            run.total_cost = 0.0;

            for(var j = 0; j < run.runs.length; j += 1) {
                run.total_cost += run.runs[j].cost;
            }
        }
    },
    sorts: {
        byName: function (a, b) {
            return d3.ascending(a.instance, b.instance);
        },
        byTotalCost: function (a, b) {
            return d3.descending(a.total_cost, b.total_cost);
        },
        bySimilarity: function (a, b) {
            var ai = borgView.instances.indexOf(a.instance);
            var bi = borgView.instances.indexOf(b.instance);
            var ti = borgView.top_instance_index;

            return d3.descending(borgView.similarity[ti][ai], borgView.similarity[ti][bi]);
        }
    },
    reorder: function(sort) {
        console.log("reordering by " + sort);

        borgView.sort = sort;

        // XXX this top-instance stuff is obviously goofy
        var runsRows = d3.selectAll("#runs-list .instance-runs");

        //console.log(runsRows[0]);

        borgView.top_instance_index = borgView.instances.indexOf(runsRows[0][0].__data__.instance);

        runsRows.sort(sort);

        borgView.top_instance_index = borgView.instances.indexOf(runsRows[0][0].__data__.instance);
    },
    renderTabular: function() {
        // render column headings
        d3.select("#runs-table > header")
            .selectAll(".solver-name-outer")
            .data(borgView.solvers)
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
                .data(borgView.runs)
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
            .attr("title", function (datum) { return datum.cost; })
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
        borgView.reorder(borgView.sorts.byName);

        // set up overall UI
        $("#row-order-select")
            .selectmenu({ style: "dropdown" })
            .change(function(event, ui) {
                borgView.reorder(borgView.sorts[$(this).val()]);
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

                borgView.reorder(borgView.sort);
            });

        // handle row-click events
        $("body")
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
    },
    renderProjection: function() {
        borgView.preprocess(); // XXX

        var $projection = $("#instance-projection");
        var xScale = d3.scale.linear().domain([-1, 1]).range([0, $projection.innerWidth()]);
        var yScale = d3.scale.linear().domain([-1, 1]).range([0, $projection.innerHeight()]);
        var runsRows = 
            d3.select("#instance-projection")
                .selectAll("circle")
                .data(borgView.projection)
                .enter()
                .append("svg:circle")
                .attr("cx", function(d) { return xScale(d[0]); })
                .attr("cy", function(d) { return yScale(d[1]); })
                .attr("r", 8);
    },
    loadThenRenderDefault: function() {
        var renderDefault = function() {
            d3.select("#data-sets-section > nav")
                .selectAll("a")
                .data(borgView.categories)
                .enter()
                .append("a")
                .attr("href", function(d) { return d.path; })
                .text(function(d) { return d.name; });

            $("#data-sets-section > nav > a")
                .click(function (event) {
                    event.preventDefault();

                    borgView.loadThenRenderCategory(event.target.__data__.path);
                });

            borgView.loadThenRenderCategory(borgView.categories[0].path);
        };

        borgView.loadThenRender(
            [
                {path: "categories.json", name: "categories"},
            ],
            renderDefault
            );
    },
    loadThenRenderCategory: function(path) {
        if (borgView.currentCategory !== path) {
            if (borgView.currentCategory !== undefined) {
                window.history.pushState(null, null, path);
            }
            else {
                window.history.pushState(null, null, "categories/" + path);
            }

            d3.selectAll("#visualization > svg *").remove();

            borgView.currentCategory = path;

            console.log("path is " + path);

            borgView.loadThenRender(
                [
                    {path: "../data/" + path + "/runs.json", name: "runs"},
                    {path: "../data/" + path + "/solvers.json", name: "solvers"},
                    {path: "../data/" + path + "/instances.json", name: "instances"},
                    {path: "../data/" + path + "/projection.json", name: "projection"}
                    //{path: "data/similarity.json", name: "similarity"}
                ],
                borgView.renderProjection
                );
        }
    },
    loadThenRender: function(requests, renderMethod) {
        var loaded = 0;
        var makeHandler = function(request) {
            return function(fetched) {
                borgView[request.name] = fetched;

                loaded += 1;

                console.log("loaded " + request.path + " (" + loaded + " of " + requests.length + ")");

                if(loaded == requests.length) {
                    //borgView.renderTabular();
                    //borgView.renderProjection();
                    renderMethod();
                }
            };
        };

        for(var i = 0; i < requests.length; i += 1) {
            var request = requests[i];

            d3.json(request.path, makeHandler(request));
        }
    }
};

window.onpopstate = function(event) {
    // XXX
    //console.log("state change!");
    //console.log(event);
};

borgView.loadThenRenderDefault();

