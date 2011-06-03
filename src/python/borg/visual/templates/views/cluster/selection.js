//
// CLUSTER VIEW SELECTIONS
//

bv.views.cluster.selection = {
    colors: d3.scale.category10(),
    visible: true,
    ready: false
};

bv.views.cluster.selection.create = function(view, point) {
    var selection = Object.create(bv.views.cluster.selection);

    selection.view = view;
    selection.p0 = point;

    return selection;
};

bv.views.cluster.selection.reify = function(number, title) {
    var $this = $(this);

    this.number = number;
    this.color = this.colors(this.number % 10);

    if(title === undefined) {
        this.titleMarkup = 'Region #%s'.format(this.number);
    }
    else {
        this.titleMarkup = title;
    }

    return this;
};

bv.views.cluster.selection.area = function() {
    if(this.p2 === undefined) {
        return 0.0;
    }
    else {
        return (this.p2.x - this.p1.x) * (this.p2.y - this.p1.y);
    }
};

bv.views.cluster.selection.update = function(q) {
    this.p1 = {x: Math.min(this.p0.x, q.x), y: Math.min(this.p0.y, q.y)};
    this.p2 = {x: Math.max(this.p0.x, q.x), y: Math.max(this.p0.y, q.y)};

    $(this).trigger("moved");

    return this;
};

bv.views.cluster.selection.toggleVisible = function() {
    this.visible = !this.visible;

    return this;
};

bv.views.cluster.selection.instances = function() {
    if(this.p0 === undefined) {
        return this.view.instances;
    }
    else {
        var this_ = this;

        return this.view.instances.filter(function(instance) {
            var cx = instance.node.cx.animVal.value;
            var cy = instance.node.cy.animVal.value;

            return true &&
                cx >= this_.p1.x && cx <= this_.p2.x &&
                cy >= this_.p1.y && cy <= this_.p2.y;
        });
    }
};

