//
// INTERFACE GLUE
//

var viewFactories = [
    {name: "cluster", text: "Cluster View", build: newClusterView},
    {name: "table", text: "Table View", build: newTableView},
];
var ui = {
    view: null,
    viewFactory: viewFactories[0],
    category: null,
    resources: [{name: "categories", path: "categories.json"}]
};

ui.resource = function(name, filename) {
    return {name: name, path: "data/" + this.category.path + "/" + filename};
};

ui.loaded = function() {
    // enable change-category links
    d3.select("#change-data > nav")
        .selectAll("a")
        .data(this.categories)
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
    this.changeCategory(this.categories[0]);
};

ui.changeCategory = function(category) {
    this.category = category;

    this.changeView(this.viewFactory);
};

ui.changeView = function(factory) {
    console.log("changing to category " + this.category.name + ", " + factory.name + " view");

    // update our location
    window.history.pushState(null, null, "ui/" + this.category.path + "/" + factory.name);

    // prepare to unload old view
    var callback = undefined;

    if(this.view !== null) {
        var old = this.view;

        callback = function() { old.unload(); };
    }

    // switch to the new view
    this.viewFactory = factory;
    this.view = factory.build(this);

    load(this.view, callback);
};

