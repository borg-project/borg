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

