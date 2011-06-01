//
// CLUSTER VIEW
//

bv.views.cluster = {
    name: "cluster",
    description: "Cluster View",
    plots: {}
};

bv.views.cluster.create = function() {
    var view = Object.create(bv.views.cluster);

    view.resourcesRequested = [
        bv.ui.request("runs", "runs.json"),
        bv.ui.request("solvers", "solvers.json"),
        bv.ui.request("instances", "instances.json"),
        bv.ui.request("membership", "membership.json"),
        bv.ui.request("projection", "projection.json")
    ];
    view.resources = {};
    view.selections = bv.list.create();

    $(view).bind("resources-loaded", function() {
        view.initialize();
    });

    bv.load(view);

    return view;
};

bv.views.cluster.initialize = function() {
    var this_ = this;
    var $this = $(this);

    // instances
    var instancesByName = {};

    this.instances = this.resources.instances.map(function(name, i) {
        var instance = {
            name: name,
            x: this_.resources.projection[i][0],
            y: this_.resources.projection[i][1],
            membership: this_.resources.membership[i],
            dominant: 0,
            bars: {}
        };

        for(var j = 0; j < instance.membership.length; j += 1) {
            if(instance.membership[j] > instance.membership[instance.dominant]) {
                instance.dominant = j;
            }
        }

        instancesByName[name] = instance;

        return instance;
    });

    this.resources.runs.forEach(function(runsOn) {
        instancesByName[runsOn.instance].runs = runsOn.runs;
    });

    // miscellaneous data
    this.nodes = {};
    this.shapePaths = [
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

    // start things moving
    this.prepareProjection();
    this.prepareSelections();
    this.preparePlot();
};

bv.views.cluster.prepareProjection = function() {
    // prepare projection area
    var $projection = $('<svg id="cluster-view"></svg>').appendTo("#display");
    var dprojection = d3.select($projection.get(0));
    var ddefs = dprojection.append("svg:defs");

    dprojection
        .append("svg:rect")
        .attr("id", "cluster-view-border")
        .attr("x", 1)
        .attr("y", 1)
        .attr("height", $projection.innerHeight() - 2)
        .attr("width", $projection.innerWidth() - 2)
        .style("stroke", this.selection.colors.range()[0]);

    // generate node shapes
    ddefs
        .selectAll("path")
        .data(this.shapePaths)
        .enter()
        .append("svg:path")
        .attr("id", function(d, i) { return "node-shape-%s".format(i); })
        .attr("d", function(d) { return d; });

    // initialize projection
    this.nodes.projection = dprojection.node();

    this.updateProjection();
};

bv.views.cluster.updateProjection = function() {
    // mise en place
    var this_ = this;
    var dprojection = d3.select(this.nodes.projection);
    var $projection = $(this.nodes.projection);

    // prepare scales
    var xScale =
        d3.scale.linear()
            .domain([
                d3.min(this.instances, function(d) { return d.x; }),
                d3.max(this.instances, function(d) { return d.x; })
            ])
            .rangeRound([16, $projection.innerWidth() - 16]);
    var yScale =
        d3.scale.linear()
            .domain([
                d3.min(this.instances, function(d) { return d.y; }),
                d3.max(this.instances, function(d) { return d.y; })
            ])
            .rangeRound([16, $projection.innerHeight() - 16]);

    // add instance nodes
    var denter =
        dprojection
            .selectAll(".instance-point")
            .data(this.instances)
            .enter();

    denter
        .append("svg:use")
        .attr("xlink:href", function(d) {
            return "#node-shape-%s".format(d.dominant % this_.shapePaths.length);
        })
        .attr("class", "instance-point")
        .attr("x", function(d) { return xScale(d.x) + 0.5; })
        .attr("y", function(d) { return yScale(d.y) + 0.5; })
        .attr("height", 10)
        .attr("width", 10)
        .on("mouseover", function(d) {
            d3.select(this).classed("highlighted", true);
            d3.select(d.label).attr("display", "block");
        })
        .on("mouseout", function(d) {
            d3.select(this).classed("highlighted", false);
            d3.select(d.label).attr("display", "none");
        })
        .each(function(d) {
            d.node = this;
        });

    denter
        .append("svg:text")
        .classed("instance-label", true)
        .each(function(d) {
            d.label = this;

            var right = d3.select(d.node).attr("x") > $projection.innerWidth() / 2;

            d3.select(this)
                .attr("x", xScale(d.x) + (right ? -12.5 : 12.5))
                .attr("y", yScale(d.y) + 0.5)
                .attr("display", "none")
                .style("text-anchor", right ? "end" : "start")
                .text(d.name);
        });
};

bv.views.cluster.prepareSelections = function() {
    // prepare the selection list
    var this_ = this;

    $("<p>Selections:</p>")
        .appendTo("#configuration-section > div")
        .append('<ul id="selections-list"></ul>');

    // the current selection, if any
    var selection = null;
    var count = 0;
    var finished = function() {
        if(selection !== null) {
            $(this_.selections).trigger("changed");

            selection = null;
        }
    };

    $("#cluster-view")
        .mousedown(function(event) {
            if(selection === null) {
                selection = this_.selection.create(this_, {x: event.layerX, y: event.layerY});
            }
            else {
                finished();
            }
        })
        .mousemove(function(event) {
            if(selection !== null) {
                selection.update({x: event.layerX, y: event.layerY});

                if(selection.number === undefined && selection.area() >= 25) {
                    selection.reify(count += 1);

                    this_.selections.add(selection);
                }
            }
        })
        .mouseup(function(event) {
            finished();
        });

    // update on changes
    $(this.selections).bind("changed", function() { this_.updateSelections(); });

    // add the initial "everything" selection
    var everything = this.selection.create(this).reify(0, "All Instances");

    this.selections.add(everything);
};

bv.views.cluster.updateSelections = function() {
    var this_ = this;

    // update selection list
    var ditems = 
        d3.select("#selections-list")
            .selectAll("li")
            .data(this.selections.get(), function(d) { return d.number; });

    var added =
        ditems
            .enter()
            .append("li")
            .each(function(d) {
                var $d = $(d);

                $(this)
                    .mouseenter(function() { $d.trigger("highlighted", [true]); })
                    .mouseleave(function() { $d.trigger("highlighted", [false]); });
            });

    ditems.exit().remove();

    added
        .append("span")
        .attr("title", "Toggle")
        .attr("class", "ui-icon ui-icon-pause")
        .on("click", function(d) {
            // switch the icon and text
            $(this)
                .toggleClass("ui-icon-pause")
                .toggleClass("ui-icon-play")
                .parent()
                .toggleClass("hidden");

            // hide the region
            d.toggleVisible();

            $(this_.selections).trigger("changed");
        });

    added
        .append("span")
        .filter(function(d) { return d.number > 0; })
        .attr("title", "Remove")
        .attr("class", "ui-icon ui-icon-close")
        .on("click", function(d) { this_.selections.remove(d); });

    added
        .append("span")
        .html(function(d) { return d.titleMarkup; })
        .style("color", function(d) { return d.color; });

    // update selection rectangles
    var visible = this.selections.get().filter(function(d) { return d.visible && d.p2 !== undefined; });
    var dboxes =
        d3.select("#cluster-view")
            .selectAll(".selection-region")
            .data(visible, function(d) { return d.number; });

    dboxes
        .enter()
        .append("svg:rect")
        .classed("selection-region", true)
        .attr("x", function(d) { return d.p1.x; })
        .attr("y", function(d) { return d.p1.y; })
        .attr("height", function(d) { return d.p2.y - d.p1.y; })
        .attr("width", function(d) { return d.p2.x - d.p1.x; })
        .style("fill", function(d) { return d.color; })
        .style("stroke", function(d) { return d.color; })
        .on("mouseover", function(d) { $(d).trigger("highlighted", [true]); })
        .on("mouseout", function(d) { $(d).trigger("highlighted", [false]); })
        .each(function(d) {
            var dthis = d3.select(this);

            this.highlighted = function(e, state) {
                dthis.classed("highlighted", state);
            };
            this.moved = function() {
                dthis
                    .attr("x", d.p1.x)
                    .attr("y", d.p1.y)
                    .attr("height", d.p2.y - d.p1.y)
                    .attr("width", d.p2.x - d.p1.x);
            };

            $(d)
                .bind("highlighted", this.highlighted)
                .bind("moved", this.moved);
        });

    dboxes
        .exit()
        .each(function(d) {
            $(d)
                .unbind("highlighted", this.highlighted)
                .unbind("moved", this.moved);
        })
        .remove();
};

bv.views.cluster.preparePlot = function() {
    this.nodes.configuration = $("#configuration-section > div").get(0);
    this.plot = this.plots.success.create(this, this.nodes.configuration);
};

bv.views.cluster.destroy = function() {
    $("#display").empty();
    $("#density-plot-area").remove();
    $("#configuration-section > div").empty();
};

{% include "views/cluster/selection.js" %}
{% include "views/cluster/cost_plot.js" %}
{% include "views/cluster/success_plot.js" %}

