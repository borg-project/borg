"use strict";

$(function() {

{% include "src.js/general.js" %}
{#{% include "src.js/table.js" %}#}
{% include "src.js/cluster.js" %}
{% include "src.js/interface.js" %}

load(ui);

});

