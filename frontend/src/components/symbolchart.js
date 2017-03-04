/* eslint-disable react/forbid-prop-types */

import React, { PropTypes } from 'react'
import { utcParse } from 'd3-time-format'
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
                height={370}
                margin={{ left: 70, right: 70, top: 20, bottom: 30 }}
                type='svg'
                pointsPerPxThreshold={1}
                seriesName={symbol}
                data={trades}
                xAccessor={d => d.datetime}
                xScaleProvider={discontinuousTimeScaleProvider}
            >
                <Chart id={0} yExtents={d => d.price}>
                    <XAxis axisAt='bottom' orient='bottom' ticks={10} tickStroke='#EEEEEE' />
                    <YAxis axisAt='left' orient='left' ticket={6} tickStroke='#EEEEEE' />
                    <LineSeries
                        yAccessor={d => d.price}
                        stroke='#74d400'
                        strokeDasharray='Dot'
                    />
                    <ScatterSeries
                        yAccessor={d => d.price}
                        marker={CircleMarker}
                        markerProps={{ width: 6, stroke: "#4c8a02", fill: "#74d100" }}
                    />
                </Chart>
            </ChartCanvas>
        )
    }
}

export default fitWidth(SymbolChart)
