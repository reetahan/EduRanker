// derivative source: https://observablehq.com/@korashughes/lifelane

// ** adding geographic shape
const data_dir = "nyu-2451-34509-geojson.json";

// Define a light gray background color to be used for the SVG.
// You can change this if you want another background color.
backgroundColors = ['#dcdcdc', '#FFFFFF', '#000000', '#d4d4d4']
light_blue = "#A7C7E7"
light_pink = "#ffebef"

// Window stuff
width = 1000
height = 1000
margin = {top: 50, bottom: 50, left: 130, right: 20, title: 60}
visWidth = width - margin.left - margin.right
visHeight = 800 - margin.top - margin.bottom
font_size = [] // small, med, large
// const margin = {top: 40, right: 0, bottom: 0, left: 40};
// const visWidth = 900 - margin.left - margin.right;
// const visHeight = 900 - margin.top - margin.bottom;

// load geo data
d3.json(data_dir, function(nycGeo) {
	console.log(nycGeo)

	zipToGeo = d3.index(nycGeo.features, d => d.properties.zcta)
	mapZipCodes = new Set(nycGeo.features.map(d => d.properties.zcta))
	emsToColor = d3.scaleSequential([0, maxEMS], d3.interpolateReds)

	const myfont = {type: "Times New Roman", small: 14, med: 20, large: 30};
	
	document.getElementById("intro").innerHTML = "Basic Header Test\n";

	const svg = d3.create('svg')
		.attr('width', visWidth + margin.left + margin.right)
		.attr('height', visHeight + margin.top + margin.bottom);
	
	const g = svg.append("g")
		.attr("transform", `translate(${margin.left}, ${margin.top})`);
	
	// add title
	g.append("text")
		.attr("x", visWidth / 2)
		.attr("y", -margin.top + 5)
		.attr("text-anchor", "middle")
		.attr("dominant-baseline", "hanging")
		.attr("font-family", "sans-serif")
		.attr("font-size", "20px")
		.text("Test Title");

	// projecting shapes
	const projection =  d3.geoAlbers()
	.fitSize([visWidth, visHeight], nycGeo);

	const path = d3.geoPath().projection(projection);

	// adding features to shapes
	g.selectAll("path")
	.data(nycGeo.features)
	.enter().append("path")
	.attr("d", path)
	.style("fill", "red")
	.style("stroke", "grey")
	// .style("stroke-width", .05)
	.attr("id", "nycPath");

	// *****GRAPH 1*****
	// svg1 = d3.select("body")
	// 	.append("svg")
	// 	.attr('width', visWidth + margin.left + margin.right)
	//  	.attr('height', visHeight + margin.top + margin.bottom)
	// 	.attr("font-size", myfont.small)
	// 	.attr("style", "background-color:" + backgroundColors[0]);
	// // Rectangles		
	// const g1 = svg1.append("g")
	// 	.attr('transform', `translate(${margin.left}, ${margin.top})`);
	// // bind our data to rectangles
	// g1.selectAll('rect').data(unemployment).join('rect')
	// 	// set attributes for each bar
	// 	.attr('x', 0)
	// 	.attr('y', d => stateScale(d.state))
	// 	.attr('width', d => rateScale(d.rate))
	// 	.attr('height', stateScale.bandwidth())
	// 	.attr('fill', d => rateColor(d.rate));

	// // add a group for the y-axis
	// g1.append('g')
	// 	.call(yAxis)
	// 	.call(g => g.select('.domain').remove())
	// // add a label for the y-axis
	// .append('text')
	// 	.attr('fill', 'black')
	// 	.attr('x', -80)
	// 	.attr('y', visHeight/2)
	// 	.attr("font-family", myfont.type)
	// 	.attr("font-size", myfont.med)
	// 	.text("State");
	// // add a group for the x-axis
	// g1.append('g')
	// 	// we have to move this group down to the bottom of the vis
	// 	.attr('transform', `translate(0, ${visHeight})`)
	// 	.call(xAxis)
	// 	.call(g => g.select('.domain').remove())
	// // add a label for the x-axis
	// .append('text')
	// 	.attr('fill', 'black')
	// 	.attr('x', visWidth/2)
	// 	.attr('y', 35)
	// 	.attr("font-family", myfont.type)
	// 	.attr("font-size", myfont.med)
	// 	.text("Unemployment Rate");
	// title
	svg1.append('text')
		.attr('fill', 'black')
		.attr('x', visWidth/2 - margin.title)
		.attr('y', 40)
		.attr("font-family", myfont.type)
		.attr("font-size", myfont.large)
		.text("Test Title 2");

}, function(reason) {
	console.log(reason); // Error!
	d3.select("body")
		.append("p")
		.text("Could not load CSV data set. See console for more information.");
});


