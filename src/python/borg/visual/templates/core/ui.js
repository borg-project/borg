//
// INTERFACE GLUE
//

bv.ui = {
    view: null,
    category: null,
    resourcesRequested: [{name: "categories", path: "categories.json"}],
    resources: {}
};

bv.ui.request = function(name, filename) {
    return {name: name, path: "data/" + this.category.path + "/" + filename};
};

bv.ui.changeCategory = function(category) {
    this.category = category;

    this.changeView(this.view);
};

bv.ui.changeView = function(factory) {
    console.log("changing to category " + this.category.name + ", " + factory.name + " view");

    // update our location
    window.history.pushState(null, null, "ui/" + this.category.path + "/" + factory.name);

    // switch to the new view
    if(this.view !== null) {
        this.view.unload();
    }

    this.view = factory.create();
};

$(bv.ui).bind("resources-loaded", function() {
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

            bv.ui.changeCategory(this.__data__);
        });

    // enable change-view links
    var views = Object.keys(bv.views).map(function(name) { return bv.views[name]; });

    d3.select("#change-view > nav")
        .selectAll("a")
        .data(views)
        .enter()
        .append("a")
        .attr("href", "")
        .text(function(d) { return d.description; })
        .on("click", function() {
            d3.event.preventDefault();

            bv.ui.changeView(this.__data__);
        });

    // get the ball rolling
    this.category = this.resources.categories[0];

    this.changeView(bv.views.cluster);
});

//self.location = function () {
    //// compute our root URL
    //var pieces = window.location.href.split("/");
    //var components = {};

    //for(var i = pieces.length - 1; i >= 0; i -= 1) {
        //if (pieces[i] === "borgview") {
            //components.root = pieces.slice(0, i + 1).join("/");

            //break;
        //}
    //}

    //if(components.root === undefined) {
        //throw {name: "AssertionError", message: "could not establish root URL"};
    //}

    //return components;
//};

bv.load(bv.ui);

