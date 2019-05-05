google.load('visualization', '1', {
  'packages': ['corechart']
});
google.setOnLoadCallback(init);

var Chart = React.createClass({
  displayName: "Chart",
  getInitialState: function() {
    return {
      data: this.getData()
    };
  },
  render: function() {
    return React.DOM.div({
      id: this.props.graphName,
      style: {
        width: '800px',
        height: '500px'
      }
    });
  },
  componentDidMount: function() {
    this.draw();
    this.interval = setInterval(this.update, 4000);
  },
  update: function(){
    this.setState({
      data : this.getData()
    });
    this.draw();
  },
  draw: function() {
    var data = this.state.data;
    var options = {
      title: 'Chart showing random data auto updating',
      vAxis: {
        viewWindow: {
          max: 40,
          min: 0
        }
      }
    };
    var element = document.getElementById(this.props.graphName);
    var chart = new google.visualization.LineChart(element);
    chart.draw(data, options);
  },
  getData: function() {
    return google.visualization.arrayToDataTable([
      ['Random X', 'Random Y'],
      ['1', Math.floor(Math.random() * 40) + 1],
      ['2', Math.floor(Math.random() * 40) + 1],
      ['3', Math.floor(Math.random() * 40) + 1],
      ['4', Math.floor(Math.random() * 40) + 1],
      ['5', Math.floor(Math.random() * 40) + 1],
    ]);
  }
});
function init() {
  React.render(React.createElement(Chart, {
    graphName: "line"
  }), document.body);
}