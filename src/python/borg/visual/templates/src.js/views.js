//
// VIEW BASE
//

var newView = function(requests) {
    return {
        resourcesRequested: requests,
        resources: {}
    };
};

//
// INDIVIDUAL VIEWS
//

{% include "src.js/views/table.js" %}
{% include "src.js/views/cluster.js" %}

