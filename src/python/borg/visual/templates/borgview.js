// keep the table header in view
$(function() {
    var $tableHeader = $("div#table-header");
    var headerTop = $tableHeader.offset().top;

    $(window).scroll(function() {
        if($(document).scrollTop() > headerTop) {
            if(!$tableHeader.data("overlaid")) {
                var tableWidth = $("div#introduction").outerWidth();

                $("div#introduction").css("margin-bottom", $tableHeader.outerHeight());

                $tableHeader
                    .css("position", "fixed")
                    .width(tableWidth - 4) // XXX hard-coded constant
                    .data("overlaid", true);
            }
        }
        else {
            if($tableHeader.data("overlaid")) {
                $tableHeader
                    .css("position", "static")
                    .data("overlaid", false);

                $("div#introduction").css("margin-bottom", "");
            }
        }
    });
});

borgView = { loaded: 0 };

borgView.render = function() {
    // render column headings
    d3.select("#table-header")
        .selectAll(".solver-name-outer")
        .data(borgView.solvers)
        .enter()
        .append("div")
        .attr("class", "solver-name-outer")
        .append("div")
        .attr("class", "solver-name")
        .text(function(name) { return name; });

    var nameWidths = $("div.solver-name-outer").map(function(i, o) { return $(o).outerWidth(); });

    $("div#table-header").height(d3.max(nameWidths));
    $("div.solver-name-outer").width("25px");
    $("div.solver-name").css({
        "-webkit-transform": "rotate(90deg)",
        "-moz-transform": "rotate(90deg)",
        "-ms-transform": "rotate(90deg)",
        "-o-transform": "rotate(90deg)",
        "transform": "rotate(90deg)"
    });

    // compute run costs
    for(var i = 0; i < borgView.runs.length; i += 1) {
        var run = borgView.runs[i];

        run.total_cost = 0.0;

        for(var j = 0; j < run.runs.length; j += 1) {
            run.total_cost += run.runs[j].cost;
        }
    }

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

    // provide various sort orders
    var sorts = {
        byName: function (a, b) {
            return d3.ascending(a.instance, b.instance);
        },
        byTotalCost: function (a, b) {
            return d3.descending(a.total_cost, b.total_cost);
        },
        bySimilarity: function (a, b) {
            //console.log("comparing " + a.instance + " and " + b.instance)
            //console.log(runsRows.call());
            var ai = borgView.instances.indexOf(a.instance);
            var bi = borgView.instances.indexOf(b.instance);

            return d3.descending(borgView.similarity[sorts.ti][ai], borgView.similarity[sorts.ti][bi]);
        }
    };

    runsRows.sort(sorts.byName);

    // set up overall UI
    $("#row-order-select")
        .selectmenu({ style: "dropdown" })
        .change(function(event, ui) {
            sorts.ti = borgView.instances.indexOf(runsRows[0][0].__data__.instance);

            runsRows.sort(sorts[$(this).val()]);
        });

    // set up row interface UI
    $(window).click(function() {  });
    $("button.raise-button")
        .button({
            icons: { primary: "ui-icon-arrowthickstop-1-n" },
            text: false
        })
        .click(function() {
        }
    );

    // handle row-click events
    $("body")
        .click(function() {
            $("li.ui-selected").removeClass("ui-selected");
            $("div#row-interface").hide();
        }
    );
    $("li.instance-runs")
        .mousedown(function(event) {
            //$(this)
                //.addClass("ui-selecting")
                //.siblings()
                //.removeClass("ui-selected")
                //.removeClass("ui-selecting");
            //$("div#row-interface").hide();
        })
        .click(function() {
            $(this)
                .addClass("ui-selected")
                .siblings()
                .removeClass("ui-selected");

            $("div#row-interface")
                .css({ top: $(this).position().top + $(this).outerHeight() })
                .show();

            event.stopPropagation();
        }
    );
};

borgView.loadThenRender = function(requests) {
    var makeHandler = function(request) {
        return function(fetched) {
            borgView[request.name] = fetched;
            borgView.loaded += 1;
            console.log("loaded " + request.path);

            if(borgView.loaded == requests.length)
                borgView.render();
        };
    };

    for(var i = 0; i < requests.length; i += 1) {
        var request = requests[i];

        d3.json(request.path, makeHandler(request));
    }
};

borgView.loadThenRender([
    {path: "data/runs.json", name: "runs"},
    {path: "data/solvers.json", name: "solvers"},
    {path: "data/instances.json", name: "instances"},
    {path: "data/similarity.json", name: "similarity"}
]);

