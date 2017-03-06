/* eslint-disable react/forbid-prop-types, react/require-default-props */

import React, { PropTypes } from 'react'
import { format } from 'd3-format'
import { GenericChartComponent, tooltip, utils } from 'react-stockcharts'

const { ToolTipText } = tooltip
const { functor } = utils

/* tooltip text components */
const TooltipTSpanLabel = props => (
    <tspan className='purplechart-tooltip-label' textAnchor='end' fill='#999999' {...props}>{props.children}</tspan>
)

TooltipTSpanLabel.propTypes = {
    children: PropTypes.node.isRequired,
}

const TooltipTSpanValue = props => (
    <tspan className='purplechart-tooltip-value' fill='#333333' {...props}>{props.children}</tspan>
)

TooltipTSpanValue.propTypes = {
    children: PropTypes.node.isRequired,
}


class ChartTooltipElement extends React.Component {
    render() {
        const { fontFamily, fontSize, displayName, value, id, height, x } = this.props
        const translate = `translate(${x}, ${id * height})`
        return (
            <g transform={translate}>
                <ToolTipText
                    x={5} y={0}
                    fontFamily={fontFamily}
                    fontSize={fontSize}
                >
                    <TooltipTSpanLabel>{displayName}</TooltipTSpanLabel>
                    <TooltipTSpanValue x='28' dy='0'>{value}</TooltipTSpanValue>
                </ToolTipText>
            </g>
        )
    }
}

ChartTooltipElement.propTypes = {
    id: PropTypes.number.isRequired,
    height: PropTypes.number,
    x: PropTypes.number,
    displayName: PropTypes.string.isRequired,
    value: PropTypes.string.isRequired,
    fontFamily: PropTypes.string,
    fontSize: PropTypes.number,
    forChart: PropTypes.number.isRequired,
    options: PropTypes.object.isRequired,
}

ChartTooltipElement.defaultProps = {
    height: 17,
    options: {},
    x: 15
}

class ChartTooltip extends React.Component {
    constructor(props) {
        super(props)
        this.renderSVG = this.renderSVG.bind(this)
    }

    renderSVG(moreProps) {
        const { chartId, chartConfig, currentItem } = moreProps

        const { className, width, fontFamily, fontSize, origin: originProp } = this.props
        const { chartConfig: { height } } = moreProps

        const config = chartConfig

        const origin = functor(originProp)
        const [x, y] = origin(width, height)
        const [ox, oy] = config.origin

        /* small hack to get background height */
        const elCount = 4
        const elHeight = 15
        const elMargin = 2
        const bgMinHeight = 18

        const rectHeight = (elCount * elHeight) + (elCount - 1) * elMargin + bgMinHeight // eslint-disable-line

        const price = currentItem ? format('.2f')(currentItem.price).toString() : 'n/a'
        const volume = currentItem ? format('.4s')(currentItem.size).toString() : 'n/a'
        const change = 'n/a' // TODO : implement
        const flagged = currentItem ? currentItem.flagged.toString() : 'n/a'

        return (
            <g
                transform={`translate(${ox + x}, ${oy + y})`}
                className={className}
                ref={(el) => { this.tooltipGroup = el }}
            >
                <rect
                    x={-33}
                    y={-20}
                    rx='4'
                    ry='4'
                    width={140}
                    height={rectHeight}
                    fill='#FFFFFF'
                    stroke='#555'
                    opacity='0.9'
                />
                <ChartTooltipElement
                    id={0}
                    displayName='Price'
                    value={price}
                    forChart={chartId}
                    fontFamily={fontFamily}
                    fontSize={fontSize}
                />
                <ChartTooltipElement
                    id={1}
                    displayName='Vol'
                    value={volume}
                    forChart={chartId}
                    fontFamily={fontFamily}
                    fontSize={fontSize}
                />
                <ChartTooltipElement
                    id={2}
                    displayName='% Chg'
                    value={change}
                    forChart={chartId}
                    fontFamily={fontFamily}
                    fontSize={fontSize}
                />
                <ChartTooltipElement
                    id={3}
                    displayName='Flag'
                    value={flagged}
                    fontFamily={fontFamily}
                    fontSize={fontSize}
                />
            </g>
        )
    }

    render = () => (
        <GenericChartComponent
            clip={false}
            svgDraw={this.renderSVG}
            drawOnMouseMove
        />
    )
}

ChartTooltip.propTypes = {
    className: PropTypes.string,
    origin: PropTypes.array.isRequired,
    fontFamily: PropTypes.string,
    fontSize: PropTypes.number,
    width: PropTypes.number,
    forDataSeries: PropTypes.arrayOf(PropTypes.number),
}

ChartTooltip.defaultProps = {
    className: 'react-stockcharts-toottip react-stockcharts-moving-average-tooltip',
    displayFormat: format('.2f'),
    origin: [0, 10],
    width: 65,
    fontSize: 13,
}

export default ChartTooltip
