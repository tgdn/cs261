/* eslint-disable react/forbid-prop-types */

import React, { PropTypes } from 'react'
import { utcParse, timeFormat, scaleTime } from 'd3-time-format'
import { format, formatDefaultLocale } from 'd3-format'
import { ChartCanvas, Chart, series, scale, coordinates, tooltip, axes, helper } from 'react-stockcharts'

const { BarSeries, LineSeries, ScatterSeries, CircleMarker, SquareMarker, TriangleMarker } = series
const { discontinuousTimeScaleProvider } = scale

const { CrossHairCursor, MouseCoordinateX, MouseCoordinateY, CurrentCoordinate } = coordinates

const { MovingAverageTooltip } = tooltip
const { XAxis, YAxis } = axes
const { fitWidth } = helper

const parseDatetime = utcParse('%Y-%m-%dT%H:%M:%S.%LZ')

formatDefaultLocale({
    decimal: '.',
    thousands: ',',
    grouping: [3],
    currency: ['£', '']
})

class SymbolChart extends React.Component {
    constructor(props) {
        super(props)
        const { trades, minDatetime, maxDatetime, maxVolume } = this.parseData(this.props.trades)
        this.state = {
            trades,
            maxVolume,
            initialMinDatetime: minDatetime,
            initialMaxDatetime: maxDatetime
        }
    }

    shouldComponentUpdate(newProps, newState) {
        const latestTrade = this.state.trades[this.state.trades.length - 1].id
        const newLatestTrade = newState.trades[newState.trades.length - 1].id
        if (latestTrade !== newLatestTrade) {
            return true
        }
        return false
    }

    parseData(trades) {
        let maxVolume = 0
        trades.forEach((d) => {
            d.datetime = new Date(parseDatetime(d.datetime)) // eslint-disable-line
            if (d.size > maxVolume) {
                maxVolume = d.size
            }
        })
        const maxDatetime = trades[trades.length - 1].datetime
        let minDatetime = trades[0].datetime
        if (trades.length >= 40) {
            minDatetime = trades[trades.length - 40].datetime
        }
        return { trades, minDatetime, maxDatetime, maxVolume }
    }

    componentWillReceiveProps(newProps) {
        const { trades, minDatetime, maxDatetime, maxVolume } = this.parseData(newProps.trades)
        this.setState({
            trades,
            maxVolume,
            initialMinDatetime: minDatetime,
            initialMaxDatetime: maxDatetime,
        })
    }

    render() {
        const { symbol, width, ratio } = this.props
        const { trades, initialMinDatetime, initialMaxDatetime } = this.state

        const margin = { left: 70, right: 80, top: 20, bottom: 50 }
        const height = 600
        const gridHeight = height - margin.top - margin.bottom
        const gridWidth = width - margin.left - margin.right
        const yGrid = {
            innerTickSize: -1 * gridWidth,
            tickStrokeDasharray: 'Solid',
            tickStrokeOpacity: 0.2,
            tickStrokeWidth: 0.2
        }
        const xGrid = {
            innerTickSize: -1 * gridHeight,
            tickStrokeDasharray: 'Solid',
            tickStrokeOpacity: 0.2,
            tickStrokeWidth: 0.2
        }
        // <ScatterSeries
        //     yAccessor={d => d.price}
        //     marker={CircleMarker}
        //     markerProps={{ r: 3, stroke: '#878712', fill: '#C1C11F' }}
        // />
        return (
            <div>
                <ChartCanvas
                    ratio={ratio}
                    width={width}
                    height={height}
                    margin={margin}
                    type='svg'
                    pointsPerPxThreshold={1}
                    seriesName={symbol}
                    data={trades}
                    xAccessor={d => d.datetime}
                    xScaleProvider={discontinuousTimeScaleProvider}
                    xExtents={[initialMinDatetime, initialMaxDatetime]}
                >
                    <Chart
                        id={1}
                        yExtents={d => d.price}
                        padding={{ top: 10, bottom: 20 }}
                        height={430}
                    >
                        <XAxis
                            axisAt='bottom'
                            orient='bottom'
                            opacity={0.5}
                            showTicks={false}
                            outerTickSize={0}
                            tickStroke='#FFFFFF'
                            stroke='#FFFFFF'
                        />
                        <YAxis
                            axisAt='right'
                            orient='right'
                            ticks={6}
                            tickStroke='#FFFFFF'
                            stroke='#FFFFFF'
                        />
                        <MouseCoordinateY
                            at="right"
                            orient="right"
                            displayFormat={format('$.2f')}
                        />
                        <LineSeries
                            yAccessor={d => d.price}
                            stroke='#F4F42E'
                            strokeDasharray='Solid'
                        />
                        <CurrentCoordinate yAccessor={d => d.price} stroke='#F4F42E' />
                    </Chart>
                    <Chart
                        id={2}
                        height={100}
                        yExtents={d => [0, this.state.maxVolume * 10]}
                        origin={(w, h) => [0, h - 100]}
                    >
                        <XAxis
                            showGrid
                            axisAt='bottom'
                            orient='bottom'
                            tickStroke='#FFFFFF'
                            stroke='#FFFFFF'
                        />
                        <YAxis
                            axisAt='left'
                            orient='left'
                            ticks={5}
                            tickFormat={format('.0s')}
                            tickStroke='#FFFFFF'
                            stroke='#FFFFFF'
                        />
                        <MouseCoordinateX
                            at="bottom"
                            orient="bottom"
                            displayFormat={timeFormat('%H:%M:%S')}
                        />
                        <MouseCoordinateY
                            at="right"
                            orient="right"
                            displayFormat={format('.4s')}
                        />
                        <BarSeries
                            yAccessor={d => d.size}
                            fill='#6BA583'
                            clip={false}
                        />
                    </Chart>
                    <CrossHairCursor stroke='#BBBBBB' />
                </ChartCanvas>
            </div>
        )
    }
}

export default fitWidth(SymbolChart)
