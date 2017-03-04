/* eslint-disable react/forbid-prop-types */

import React, { PropTypes } from 'react'
import { utcParse, timeFormat } from 'd3-time-format'
import { format } from 'd3-format'
import { ChartCanvas, Chart, series, scale, coordinates, tooltip, axes, helper } from 'react-stockcharts'

const { BarSeries, LineSeries, AreaSeries, ScatterSeries, CircleMarker, SquareMarker, TriangleMarker } = series
const { discontinuousTimeScaleProvider } = scale

const { CrossHairCursor, MouseCoordinateX, MouseCoordinateY } = coordinates

const { OHLCTooltip } = tooltip
const { XAxis, YAxis } = axes
const { fitWidth } = helper

const parseDatetime = utcParse("%Y-%m-%dT%H:%M:%S.%LZ")

class SymbolChart extends React.Component {
    render() {
        const { symbol, trades, width, ratio } = this.props
        trades.forEach((d, i) => {
            d.datetime = new Date(parseDatetime(d.datetime))
        })

        return (
            <ChartCanvas
                ratio={ratio}
                width={width}
                height={400}
                margin={{ left: 50, right: 70, top: 20, bottom: 50 }}
                type='hybrid'
                pointsPerPxThreshold={1}
                seriesName={symbol}
                data={trades}
                xAccessor={d => d.datetime}
                xScaleProvider={discontinuousTimeScaleProvider}
            >
                <Chart
                    id={1}
                    yExtents={d => d.price}
                    height={400}
                    padding={{ top: 30, bottom: 30 }}
                >
                    <XAxis axisAt='bottom' orient='bottom' ticks={5} tickStroke='#EEEEEE' />
                    <YAxis axisAt='right' orient='right' ticks={6} tickStroke='#EEEEEE' />
                    <MouseCoordinateX
                        at="bottom"
                        orient="bottom"
                        displayFormat={timeFormat("%H:%M:%S.%L")}
                    />
                    <MouseCoordinateY
                        at="right"
                        orient="right"
                        displayFormat={format(".2f")}
                    />
                    <LineSeries
                        yAccessor={d => d.price}
                        stroke='#DB0000'
                        strokeDasharray='Solid'
                    />
                    <ScatterSeries
                        yAccessor={d => d.price}
                        marker={CircleMarker}
                        markerProps={{ r: 3, stroke: "#4c8a02", fill: "#74d100" }}
                    />
                </Chart>
                <Chart
                    id={2}
                    yExtents={d => d.size}
                    height={200}
                    origin={(w, h) => [0, 170]}
                >
                    <YAxis axisAt='left' orient='left' ticks={5} tickStroke='#EEEEEE' tickFormat={format(".0s")} />
                    <BarSeries
                        yAccessor={d => d.size}
                        fill='#74d400'
                    />
                </Chart>
            </ChartCanvas>
        )
    }
}

export default fitWidth(SymbolChart)
