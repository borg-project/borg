//
// WATCHED LIST
//

bv.list = {};

bv.list.create = function(array) {
    var list = Object.create(bv.list);

    if(array === undefined) {
        list.array = [];
    }
    else {
        list.array = array;
    }

    return list;
};

bv.list.get = function() {
    return this.array;
};

bv.list.add = function(object) {
    this.array.push(object);

    $(this).trigger("changed");
};

bv.list.remove = function(object) {
    var i = this.array.indexOf(object);

    if(i >= 0) {
        this.array.splice(i, 1);

        $(this).trigger("changed");
    }
};

