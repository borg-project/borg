//
// INTERFACE STATE
//

bv.state = {};

bv.state.initial = function(ui) {
    var state = Object.create(bv.state);

    state.viewFactory = bv.views.cluster;
    state.category = ui.resources.categories[0];
    state.options = {};

    return state;
};

bv.state.url = function() {
    return "ui/%s/%s".format(this.category.path, this.viewFactory.name);
};

bv.state.withCategory = function(category, options) {
    var state = Object.create(this);

    state.category = category;
    state.options = options;

    return state;
};

bv.state.withView = function(viewFactory, options) {
    var state = Object.create(this);

    state.viewFactory = viewFactory;
    state.options = options;

    return state;
};

bv.state.load = function() {
    window.history.pushState(this, null, this.url());

    this.view = this.viewFactory.create(this.options);

    return this;
};

bv.state.destroy = function() {
    this.view.destroy();
};

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

