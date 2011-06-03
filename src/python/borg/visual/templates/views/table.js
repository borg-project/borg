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
        bv.ui.request("similarity", "similarity.json")
    ];
    view.options = options === undefined ? {} : options;
    view.resources = {};
    view.nodes = {};

    $(view).bind("resources-loaded", function() { view.initialize(); });

    bv.load(view);

    return view;
};

bv.views.table.initialize = function() {
    // compute run costs
    if(this.options.instances !== undefined) {
        var subset = this.options.instances;
        var inSubset = function(r) {
            return subset.indexOf(r.instance) >= 0;
        };

        this.runs = this.resources.runs.filter(inSubset);
    }
    else {
        this.runs = this.resources.runs;
    }

    this.runs.forEach(function(runsOn) {
        runsOn.totalCost = runsOn.runs.reduce(function(a, b) { return a + b.cost; }, 0);
        runsOn.maxCost = d3.max(runsOn.runs, function(d) { return d.cost; });
    });

    this.maxCost = d3.max(this.runs, function(r) { return r.maxCost; });

    // get things moving
    this.prepareTable();
    this.prepareHeader();
};

bv.views.table.prepareTable = function() {
    // set up the DOM
    var markup = [
        '<div id="bv-runs-table">',
        '<header id="bv-runs-table-header">',
        //'    <div id="header-ui-controls">',
        //'        <!--<label for="row-order-select">Order Rows By:</label>-->',
        //'        <select id="row-order-select">',
        //'            <option value="byName">Sort by Name</option>',
        //'            <option value="byTotalCost">Sort by Total Cost</option>',
        //'            <option value="bySimilarity">Sort by Similarity</option>',
        //'        </select>',
        //'    </div>',
        '</header>',
        '<ul id="runs-list"></ul>',
        '<div id="row-interface">',
        '    <button class="raise-button">Raise</button>',
        '</div>',
        '</div>'
    ];
    var $markup = $(markup.join("\n")).appendTo("#display");

    this.nodes.table = $("#bv-runs-table").get(0);
    this.nodes.header = $("#bv-runs-table-header").get(0);
    this.nodes.runsList = $("#runs-list").get(0);

    // render runs table
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
        .selectAll()
        .data(function(d) { return [d.instance]; })
        .enter()
        .append("div")
        .classed("instance-title", true)
        .text(function(d) { return d; });

    d3runsRows
        .selectAll()
        .data(function(d) { return d.runs; })
        .enter()
        .append("div")
        .classed("instance-run", true)
        .attr("title", function(d) { return d.cost; })
        .style("background-color", function(d) {
            return d.answer !== null ? cColor(cScale(d.cost)) : "black";
        });
};

bv.views.table.prepareHeader = function() {
    // keep the table header in view
    var $table = $(this.nodes.table);
    var $header = $(this.nodes.header);
    var headerTop = $header.offset().top;
    var overlaid = false;

    $(window).scroll(function() {
        if($(document).scrollTop() > headerTop) {
            if(!overlaid) {
                var tableWidth = $table.outerWidth();

                $table.css("margin-top", $header.outerHeight());

                $header
                    .css("position", "fixed")
                    .css("top", headerTop)
                    .width(tableWidth - 4);

                overlaid = true;
            }
        }
        else {
            if(overlaid) {
                $table.css("margin-top", "");

                $header
                    .css("position", "static")
                    .css("top", "");

                overlaid = false;
            }
        }
    });

    // render column headings
    d3.select(this.nodes.header)
        .selectAll(".solver-name-outer")
        .data(this.resources.solvers)
        .enter()
        .append("div")
        .attr("class", "solver-name-outer")
        .append("div")
        .attr("class", "solver-name")
        .text(function(name) { return name; });

    var nameWidths =
        $table
            .find("div.solver-name-outer")
            .map(function(i, o) { return $(o).outerWidth(); });

    $header.height(d3.max(nameWidths));
    $("div.solver-name-outer").width("25px");
    $("div.solver-name").css({
        "-webkit-transform": "rotate(90deg)",
        "-moz-transform": "rotate(90deg)",
        "-ms-transform": "rotate(90deg)",
        "-o-transform": "rotate(90deg)",
        "transform": "rotate(90deg)"
    });

    //// set up overall UI
    //$("#row-order-select")
        //.selectmenu({ style: "dropdown" })
        //.change(function(event, ui) {
            //view.reorder(sorts[$(this).val()]);
        //});

    //// set up row interface UI
    //$("button.raise-button")
        //.button({
            //icons: { primary: "ui-icon-arrowthickstop-1-n" },
            //text: false
        //})
        //.click(function() {
            //$("li.ui-selected")
                //.detach()
                //.prependTo("#runs-list");

            //view.reorder(view.sort);
        //});

    //// handle row-click events
    //$("body") // XXX correctly remove this handler on unload
        //.click(function() {
            //$("li.ui-selected").removeClass("ui-selected");
            //$("#row-interface").hide();
        //});
    //$("li.instance-runs")
        //.click(function() {
            //$(this)
                //.addClass("ui-selected")
                //.siblings()
                //.removeClass("ui-selected");

            //$("#row-interface")
                //.css({ top: $(this).position().top + $(this).outerHeight() })
                //.show();

            //event.stopPropagation();
        //});

    //// by default, sort by instance name
    //this.reorder(sorts.byName);
};

//var sorts = {
    //byName: function(a, b) {
        //return d3.ascending(a.instance, b.instance);
    //},
    //byTotalCost: function(a, b) {
        //return d3.descending(a.total_cost, b.total_cost);
    //},
    //bySimilarity: function(a, b) {
        //var ai = view.instances.indexOf(a.instance);
        //var bi = view.instances.indexOf(b.instance);
        //var ti = view.top_instance_index;

        //return d3.descending(view.similarity[ti][ai], view.similarity[ti][bi]);
    //}
//};

bv.views.table.reorder = function(sort) {
    console.log("reordering by " + sort);

    this.sort = sort;

    var runsRows = d3.selectAll("#runs-list .instance-runs");

    this.top_instance_index = this.instances.indexOf(runsRows[0][0].__data__.instance);

    runsRows.sort(sort);

    this.top_instance_index = this.instances.indexOf(runsRows[0][0].__data__.instance);
};

bv.views.table.destroy = function() {
    $("#display").empty();
};

