//
// INTERFACE GLUE
//

bv.ui = {
    resourcesRequested: [{name: "categories", path: "categories.json"}],
    resources: {}
};

bv.ui.initialize = function() {
    // enable change-category links
    var this_ = this;

    d3.select("#change-data > nav")
        .selectAll("a")
        .data(this.resources.categories)
        .enter()
        .append("a")
        .attr("href", "")
        .text(function(d) { return d.name; })
        .on("click", function(d) {
            d3.event.preventDefault();

            this_.change(this_.state.withCategory(d));
        });

    // enable change-view links
    var views = d3.values(bv.views);

    d3.select("#change-view > nav")
        .selectAll("a")
        .data(views)
        .enter()
        .append("a")
        .attr("href", "")
        .text(function(d) { return d.description; })
        .on("click", function(d) {
            d3.event.preventDefault();

            this_.change(this_.state.withView(d));
        });

    // get the ball rolling
    this.change(bv.state.initial(this));
};

bv.ui.change = function(state) {
    if(this.state !== undefined) {
        this.state.destroy();
    }

    this.state = state;

    this.state.load();
};

bv.ui.request = function(name, filename) {
    return {
        name: name,
        path: "data/" + this.state.category.path + "/" + filename
    };
};

$(bv.ui).bind("resources-loaded", function() { this.initialize(); });

bv.load(bv.ui);

