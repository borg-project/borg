//
// INTERFACE GLUE
//

var viewFactories = [
    {name: "cluster", text: "Cluster View", build: newClusterView}
    //{name: "table", text: "Table View", build: newTableView},
];
var ui = {
    view: null,
    viewFactory: viewFactories[0],
    category: null,
    resourcesRequested: [{name: "categories", path: "categories.json"}],
    resources: {}
};

ui.resource = function(name, filename) {
    return {name: name, path: "data/" + this.category.path + "/" + filename};
};

ui.changeCategory = function(category) {
    this.category = category;

    this.changeView(this.viewFactory);
};

ui.changeView = function(factory) {
    console.log("changing to category " + this.category.name + ", " + factory.name + " view");

    // update our location
    window.history.pushState(null, null, "ui/" + this.category.path + "/" + factory.name);

    // switch to the new view
    if(this.view !== null) {
        this.view.unload();
    }

    this.viewFactory = factory;
    this.view = factory.build(this);

    load(this.view);
};

$(ui).bind("resources-loaded", function() {
    // enable change-category links
    d3.select("#change-data > nav")
        .selectAll("a")
        .data(this.resources.categories)
        .enter()
        .append("a")
        .attr("href", "")
        .text(function(d) { return d.name; });

    $("#change-data > nav > a")
        .click(function(event) {
            event.preventDefault();

            ui.changeCategory(this.__data__);
        });

    // enable change-view links
    d3.select("#change-view > nav")
        .selectAll("a")
        .data(viewFactories)
        .enter()
        .append("a")
        .attr("href", "")
        .text(function(d) { return d.text; });

    $("#change-view > nav > a")
        .click(function(event) {
            event.preventDefault();

            ui.changeView(this.__data__);
        });

    // get the ball rolling
    this.changeCategory(this.resources.categories[0]);
});

