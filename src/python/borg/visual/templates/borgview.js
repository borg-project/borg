//var h = 600;
//var w = 800;

//var vis = new pv.Panel()
    //.width(w)
    //.height(h)
    //.bottom(20)
    //.left(20)
    //.events("all")
    //.event("mousemove", pv.Behavior.point());

//vis.add(pv.Layout.Grid)
    //.rows([data, data])
    //.cell.add(pv.Bar)
    //.def("active", -1)
    //.fillStyle(pv.ramp("white", "black"))
    //.event("point", function() this.active(this.index).parent)
    //.event("unpoint", function() this.active(-1).parent)
    //.anchor("right")
    //.add(pv.Label)
    //.visible(function() this.anchorTarget().active() == this.index)
    //.text(function(d) d.toFixed(2));

//vis.add(pv.Rule)
    //.data(y.ticks())
    //.bottom(y)
    //.anchor("left")
    //.add(pv.Label)
    //.text(y.tickFormat);

//vis.add(pv.Rule)
    //.data(x.ticks())
    //.left(x)
    //.anchor("bottom")
    //.add(pv.Label)
    //.text(x.tickFormat);

//vis.add(pv.Dot)
    //.def("active", -1)
    //.data(data)
    //.left(function(d) x(d.x))
    //.bottom(function(d) y(d.y))
    //.strokeStyle("orange")
    //.fillStyle(function() this.strokeStyle().alpha(0.2))
    //.event("point", function() this.active(this.index).parent)
    //.event("unpoint", function() this.active(-1).parent)
    //.anchor("right")
    //.add(pv.Label)
    //.visible(function() this.anchorTarget().active() == this.index)
    //.text(function(d) d.y.toFixed(2));

//vis.render();

renderTClassGrid = function(rates_SB, names, budgets) {
    var w = 1000;
    var h = 400;

    var root = new pv.Panel()
        .top(30)
        .width(w)
        .height(h);

    rates_data = [];

    for(i in rates_SB) {
        row = [];

        for(j in rates_SB[i]) {
            row.push({
                "i": i,
                "j": j,
                "p": rates_SB[i][j],
                });
        }

        rates_data.push(row);
    }

    var grid = root
        .add(pv.Panel)
        .data(rates_data)
        .top(function(d) { return this.index * 21; });

    grid.add(pv.Bar)
        .data(function(d) { return d; })
        .width(20)
        .height(20)
        .left(function(d) { return this.index * 21; });

    grid.anchor("right")
        .add(pv.Label)
        .top(11)
        .text("foo");

    root.add(pv.Label)
        .data(budgets)
        .textAngle(-Math.PI / 2)
        .left(function(d) { return this.index * 21 + 10; });

    //var grid = vis.add(pv.Layout.Grid);

    //grid
        //.rows(rates_SK)
        //.cell.add(pv.Bar)
        //.fillStyle(pv.ramp("#EEEEEE", "black"));

    //grid.anchor("right").add(pv.Label).data([1, 2]).visible(true).width(50).text("foo");

    root.render();
};

