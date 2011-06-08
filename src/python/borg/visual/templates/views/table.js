//
// TABLE VIEW
//

bv.views.table = {
    name: "table",
    description: "Table View"
};

bv.views.table.create = function(options) {
    var view = Object.create(bv.views.table);

    view.resourcesRequested = [
        bv.ui.request("runs", "runs.json"),
        bv.ui.request("solvers", "solvers.json"),
        bv.ui.request("instances", "instances.json"),
        bv.ui.request("membership", "membership.json"),
    ];
    view.options = options === undefined ? {} : options;
    view.resources = {};
    view.nodes = {};

    return view;
};

bv.views.table.load = function() {
    var this_ = this;

    $(this).bind("resources-loaded", function() { this_.initialize(); });

    bv.load(this);

    return this;
};

bv.views.table.initialize = function() {
    // compute run costs
    if(this.options.instances !== undefined) {
        var subset = this.options.instances;
        var inSubset = function(r) { return subset.indexOf(r.instance) >= 0; };

        this.runs = this.resources.runs.filter(inSubset);
    }
    else {
        this.runs = this.resources.runs;
    }

    // compute instance information
    var this_ = this;
    var dominance = {};

    this.resources.instances.forEach(function(name, i) {
        var membership = this_.resources.membership[i];
        var dominant = 0;

        for(var j = 0; j < membership.length; j += 1) {
            if(membership[j] > membership[dominant]) {
                dominant = j;
            }
        }

        dominance[name] = dominant;
    });

    this.runs.forEach(function(runsOn) {
        runsOn.totalCost = runsOn.runs.reduce(function(a, b) { return a + b.cost; }, 0);
        runsOn.maxCost = d3.max(runsOn.runs, function(d) { return d.cost; });
        runsOn.minCost = d3.min(runsOn.runs, function(d) { return d.cost; });
        runsOn.dominant = dominance[runsOn.instance];

        runsOn.runs.forEach(function(run) {
            run.winner = run.cost === runsOn.minCost;

            if(runsOn.answer === undefined && run.answer !== null) {
                runsOn.answer = run.answer;
            }
        });
    });

    this.maxCost = d3.max(this.runs, function(r) { return r.maxCost; });

    // get things moving
    this.prepareTable();
    this.prepareHeader();
    this.prepareControls();
};

bv.views.table.prepareTable = function() {
    // set up the DOM
    var markup = [
        '<section id="bv-table-container">',
        '<div id="bv-runs-table">',
        '<header id="bv-runs-table-header">',
        '    <div id="bv-instance-name-heading">Instance Name [<span class="sat">S</span>/<span class="unsat">U</span>/?] <span class="sidescript">(Cluster #)</span></div>',
        '</header>',
        '<ul id="runs-list"></ul>',
        '<div id="row-interface">',
        '    <button class="raise-button">Raise</button>',
        '</div>',
        '</div>',
        '</section>'
    ];
    var $markup = $(markup.join("\n")).appendTo("body");

    this.nodes.container = $markup.get(0);
    this.nodes.table = $("#bv-runs-table").get(0);
    this.nodes.header = $("#bv-runs-table-header").get(0);
    this.nodes.runsList = $("#runs-list").get(0);

    // render runs table
    var this_ = this;
    var d3runsRows = 
        d3.select(this.nodes.runsList)
            .selectAll()
            .data(this.runs)
            .enter()
            .append("li")
            .classed("instance-runs", true);

    var cScale = d3.scale.linear().domain([0.0, this.maxCost]);
    var cColor = d3.interpolateRgb("rgb(237, 248, 177)", "rgb(44, 127, 184)");

    d3runsRows
        .append("div")
        .classed("instance-title", true)
        .each(function(d) {

            if(d.answer === true) {
                d3.select(this).classed("sat", true);
            }
            else if(d.answer === false) {
                d3.select(this).classed("unsat", true);
            }
            else {
                d3.select(this).classed("unknown", true);
            }
        })
        .style("margin-left", "%spx".format(this.resources.solvers.length * 25 + 20))
        .html(function(d) {
            var su = "?";

            if(d.answer === true) {
                su = "S";
            }
            else if(d.answer === false) {
                su = "U";
            }

            return '%s <span class="sidescript">(%s)</span>'.format(d.instance, d.dominant);
        });

    d3runsRows
        .selectAll()
        .data(function(d) { return d.runs; })
        .enter()
        .append("div")
        .classed("instance-run", true)
        .style("background-color", function(d) {
            return d.answer !== null ? cColor(cScale(d.cost)) : "black";
        })
        .style("left", function(d, i) { return "%spx".format(i * 25); })
        .html(function(d) { return d.winner ? '<span class="win-label">*</span>' : ""; })
        .on("mouseover", function(d) {
            d3.select(this)
                .classed("highlighted", true)
                .append("div")
                .classed("cell-label", true)
                .text(function(d) { return "%.0f s".format(d.cost); });
        })
        .on("mouseout", function(d) {
            d3.select(this)
                .classed("highlighted", false)
                .select(".cell-label")
                .remove();
        })
        .each(function(d) { d.cell = this; });
};

bv.views.table.prepareHeader = function() {
    // render column headings
    d3.select(this.nodes.header)
        .selectAll(".solver-name-outer")
        .data(this.resources.solvers)
        .enter()
        .append("div")
        .attr("class", "solver-name-outer")
        .append("div")
        .attr("class", "solver-name")
        .text(function(d) {
            var prefix = "SAT07 reference solver:";

            if(d.indexOf(prefix) >= 0) {
                return "SAT07:" + d.substr(prefix.length);
            }
            else {
                return d;
            }
        });

    var $table = $(this.nodes.table);
    var $header = $(this.nodes.header);
    var nameWidths =
        $table
            .find("div.solver-name-outer")
            .map(function(i, o) { return $(o).outerWidth(); });

    $header
        .height(d3.max(nameWidths))
        .width($table.innerWidth());

    $table.css("margin-top", $header.outerHeight());

    $("div.solver-name-outer").width("25px");
    $("div.solver-name").css({
        "-webkit-transform": "rotate(90deg)",
        "-moz-transform": "rotate(90deg)",
        "-ms-transform": "rotate(90deg)",
        "-o-transform": "rotate(90deg)",
        "transform": "rotate(90deg)"
    });
};

bv.views.table.prepareControls = function() {
    // define sort orders
    var this_ = this;
    var sorts = {
        byName: function(a, b) {
            return d3.ascending(a.instance, b.instance);
        },
        byTotalCost: function(a, b) {
            return d3.descending(a.totalCost, b.totalCost);
        },
        byCluster: function(a, b) {
            return d3.ascending(a.dominant, b.dominant);
        }
    };
    var reorder = function(sort) {
        d3.select(this_.nodes.runsList)
            .selectAll(".instance-runs")
            .sort(sort);
    };

    // set up controls
    this.nodes.configuration = $("#configuration-div").get(0);

    var markup = [
        '<p>',
        '    Sort Table Rows:',
        '    <select id="bv-table-sort-select">',
        '        <option value="byName">By Name</option>',
        '        <option value="byTotalCost">By Total Runtime</option>',
        '        <option value="byCluster">By Cluster</option>',
        '    </select>',
        '</p>'
    ];
    var $markup = $(markup.join("\n")).appendTo(this.nodes.configuration);
    var $orderSelect = $markup.find("select");

    $orderSelect
        .selectmenu({ style: "dropdown" })
        .change(function(event, ui) {
            reorder(sorts[$(this).val()]);
        });

    // by default, sort by instance name
    reorder(sorts.byName);
};

bv.views.table.destroy = function() {
    $(this.nodes.container).remove();
    $(this.nodes.configuration).empty();
};

