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
    view.drag = null;

    $(view).bind("resources-loaded", function() { view.initialize(); });

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
    this.nodes = {
        configuration: $("#configuration-section").get(0)
    };

    // start things moving
    this.prepareProjection();
    this.prepareSelections();
    this.preparePlot();
};

bv.views.cluster.prepareProjection = function() {
    // prepare projection area
    var markup = [
        '<section id="bv-projection-container">',
        '<div id="bv-projection-title">Problem Instances (MDS Projection)</div>',
        '<svg></svg>',
        '</section>'
    ];
    var $container = $(markup.join("\n")).appendTo("body");
    var $projection = $container.children("svg");
    var dprojection = d3.select($projection.get(0));

    this.nodes.container = $container.get(0);
    this.nodes.projection = dprojection.node();
    this.nodes.projectionBorder =
        dprojection
            .append("svg:rect")
            .attr("id", "bv-projection-border")
            .attr("x", 1)
            .attr("y", 1)
            .attr("height", $projection.innerHeight() - 2)
            .attr("width", $projection.innerWidth() - 2)
            .style("stroke", this.selection.colors.range()[0])
            .node();

    this.updateProjection();
};

bv.views.cluster.updateProjection = function() {
    // prepare scales
    var $projection = $(this.nodes.projection);
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
        d3.select(this.nodes.projection)
            .selectAll(".instance-point")
            .data(this.instances)
            .enter();
    var dpoints = denter.append("svg:g");

    dpoints
        .append("svg:circle")
        .attr("class", "instance-point")
        .attr("cx", function(d) { return xScale(d.x) + 0.5; })
        .attr("cy", function(d) { return yScale(d.y) + 0.5; })
        .attr("r", 10)
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
            d.rightSide = d.node.cx.animVal.value > $projection.innerWidth() / 2;
        });
    dpoints
        .append("svg:text")
        .classed("bv-instance-number", true)
        .attr("x", function(d) { return d.node.cx.animVal.value; })
        .attr("y", function(d) { return d.node.cy.animVal.value; })
        .attr("dy", "0.5ex")
        .text(function(d) { return d.dominant; });

    // add instance node labels
    var dlabels =
        denter
            .append("svg:g")
            .classed("instance-label", true)
            .attr("display", "none")
            .each(function(d) { d.label = this; });

    dlabels
        .append("svg:text")
        .attr("x", function(d) { return d.node.cx.animVal.value + (d.rightSide ? -12.5 : 12.5); })
        .attr("y", function(d) { return d.node.cy.animVal.value + 0.5; })
        .style("text-anchor", function(d) { return d.rightSide ? "end" : "start"; })
        .text(function(d) { return d.name; })
        .each(function(d) { return d.labelText = this; });
    dlabels
        .insert("svg:rect", "text")
        .attr("x", function(d) { return d.labelText.getBBox().x - 2; })
        .attr("y", function(d) { return d.labelText.getBBox().y - 2; })
        .attr("height", function(d) { return d.labelText.getBBox().height + 4; })
        .attr("width", function(d) { return d.labelText.getBBox().width + 4; });
};

bv.views.cluster.prepareSelections = function() {
    // prepare the selection list
    var this_ = this;

    this.nodes.selectionsDiv =
        $("<p>Selections:</p>")
            .appendTo(this.nodes.configuration)
            .append('<ul id="selections-list"></ul>')
            .get(0);

    // the current selection, if any
    var selection = null;
    var count = 0;
    var finished = function() {
        if(selection !== null) {
            selection.ready = true;

            selection = null;

            $(this_.selections).trigger("changed");
        }
        else if(this_.drag !== null) {
            this_.drag = null;

            $(this_.selections).trigger("changed");
        }
    };

    $(this.nodes.projection)
        .mousedown(function(event) {
            if(event.which === 1) {
                if(this_.drag === null) {
                    if(selection === null) {
                        selection = this_.selection.create(this_, {x: event.layerX, y: event.layerY});
                    }
                    else {
                        finished();
                    }
                }
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
            else if(this_.drag !== null) {
                var q = {
                    x: event.layerX - this_.drag.startX,
                    y: event.layerY - this_.drag.startY
                };

                this_.drag.startX = event.layerX;
                this_.drag.startY = event.layerY;

                this_.drag.selection.translate(q);
            }
        })
        .mouseup(function(event) {
            if(event.which === 1) {
                finished();
            }
        });

    // update on changes
    $(this.selections).bind("changed", function() { this_.updateSelections(); });

    // add the initial "everything" selection
    var everything = this.selection.create(this).reify(0, "All Instances");

    everything.ready = true;

    $(everything).bind("highlighted", function(event, state) {
        d3.select(this_.nodes.projectionBorder).classed("highlighted", state);
    });

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
        .classed("bv-selection-name", true)
        .style("color", function(d) { return d.color; })
        .html(function(d) { return d.titleMarkup; })
        .on("click", function(d) {
            var instances = d.instances().map(function(v) { return v.name; });
            var state = bv.ui.state.withView(bv.views.table, {instances: instances});

            bv.ui.change(state);
        });

    // update selection rectangles
    var visible = this.selections.get().filter(function(d) { return d.visible && d.p2 !== undefined; });
    var dboxes =
        d3.select(this.nodes.projection)
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
        .on("mouseover", function(d) {
            $(d).trigger("highlighted", [true]);
        })
        .on("mouseout", function(d) {
            $(d).trigger("highlighted", [false]);
        })
        .on("mousedown", function(d) {
            this_.drag = {
                selection: d,
                startX: d3.event.layerX,
                startY: d3.event.layerY
            };
        })
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
    // mise en place
    var this_ = this;

    // set up DOM
    this.nodes.pickSelect =
        $('<p>Chart:<select></select></p>')
            .appendTo(this.nodes.configuration)
            .find("select")
            .attr("id", "bv-cluster-chart-select")
            .get(0);
    this.nodes.plotControlsDiv =
        $("<div></div>")
            .appendTo(this.nodes.configuration)
            .get(0);
    this.nodes.chartDiv =
        d3.select("body")
            .append("div")
            .attr("id", "bv-chart-container")
            .node();
    this.nodes.chartSVG =
        d3.select(this.nodes.chartDiv)
            .append("svg:svg")
            .node();

    // plot selection
    d3.select(this.nodes.pickSelect)
        .selectAll("option")
        .data(Object.keys(this.plots))
        .enter()
        .append("option")
        .sort(function(d) { return this_.plots[d].order; })
        .attr("value", function(d) { return d; })
        .text(function(d) { return this_.plots[d].title; });

    $(this.nodes.pickSelect)
        .selectmenu({style: "dropdown"})
        .change(function(event) {
            if(this_.plot) {
                this_.plot.destroy();
            }

            var plotNodes = {
                controlsDiv: this_.nodes.plotControlsDiv,
                chartSVG: this_.nodes.chartSVG
            };
            var plotType = this_.plots[$(this).val()];

            this_.plot = plotType.create(this_, plotNodes);
        })
        .trigger("change");
};

bv.views.cluster.destroy = function() {
    this.plot.destroy();

    $(this.nodes.configuration).empty();
    $(this.nodes.container).remove();
    $(this.nodes.chartDiv).remove();
};

{% include "views/cluster/selection.js" %}
{% include "views/cluster/cost_plot.js" %}
{% include "views/cluster/success_plot.js" %}

