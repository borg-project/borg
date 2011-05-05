//(function($) {
    //// adapted from http://ajaxian.com/archives/text-overflow-for-firefox-via-jquery

    //$.fn.ellipsis = function(enableUpdating) {
		//var style = document.documentElement.style;

		//if (!('textOverflow' in style || 'OTextOverflow' in style)) {
			//return this.each(function() {
				//var el = $(this);

				//if(el.css("overflow") == "hidden"){
					//var originalText = el.html();
					//var w = el.width();

					//var t = $(this.cloneNode(true)).hide().css({
                        //'position': 'absolute',
                        //'width': 'auto',
                        //'overflow': 'visible',
                        //'max-width': 'inherit'
                    //});
					//el.after(t);

					//var text = originalText;
					//while(text.length > 0 && t.width() > el.width()){
						//text = text.substr(0, text.length - 1);
						//t.html(text + "...");
					//}
					//el.html(t.html());

					//t.remove();

					//if(enableUpdating == true){
						//var oldW = el.width();
						//setInterval(function(){
							//if(el.width() != oldW){
								//oldW = el.width();
								//el.html(originalText);
								//el.ellipsis();
							//}
						//}, 200);
					//}
				//}
			//});
		//}
        //else
            //return this;
	//};
//})(jQuery);

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
                    .width(tableWidth)
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

var renderView = function() {
    // render column headings
    d3.select("#table-header")
        .selectAll(".solver-name-outer")
        .data(borgView.solvers)
        .enter()
        .append("div")
        .attr("class", "solver-name-outer")
        .append("div")
        .attr("class", "solver-name")
        .text(function (name) { return name; });

    var nameWidths = $.map($("div.solver-name-outer"), function (o) { return $(o).outerWidth(); });

    $("div#table-header").height(d3.max(nameWidths));
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
        d3.select("#runs-table")
            .selectAll(".instance-runs")
            .data(borgView.runs)
            .enter()
            .append("div")
            .attr("class", "instance-runs");
    var maxCost = 6100.0;
    var costScale = d3.scale.linear().domain([0.0, maxCost]);
    var costColor = d3.interpolateRgb("rgb(237,248,177)", "rgb(44,127,184)");

    runsRows
        .selectAll(".instance-title")
        .data(function(datum) { return [datum.instance]; })
        .enter()
        .append("div")
        .attr("class", "instance-title")
        .text(function (title) { return title; });

    runsRows
        .selectAll(".instance-run")
        .data(function(datum) { return datum.runs; })
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
                    return "rgb(0,0,0)";
                }
            );
    };

borgView = { "loaded": 0 };

d3.json(
    "data/runs.json",
    function(runs) {
        borgView.runs = runs;
        borgView.loaded += 1;

        if(borgView.loaded >= 2)
            renderView();
    });
d3.json(
    "data/solvers.json",
    function(solvers) {
        borgView.solvers = solvers;
        borgView.loaded += 1;

        if(borgView.loaded >= 2)
            renderView();
    });

