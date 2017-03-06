/* eslint-disable react/forbid-prop-types, react/require-default-props */

import React, { PropTypes } from 'react'
import { format } from 'd3-format'
import { GenericChartComponent, tooltip, utils } from 'react-stockcharts'

const { ToolTipText, ToolTipTSpanLabel } = tooltip
const { functor } = utils

class ChartTooltipElement extends React.Component {
    render() {
        
    }
}

class ChartTooltip extends React.Component {
    constructor(props) {
        super(props)
        this.renderSVG = this.renderSVG.bind(this)
    }

    renderSVG(moreProps) {
        const { chartId, chartConfig, currentItem } = moreProps

        const { className, onClick, width, fontFamily, fontSize, origin: originProp, calculators, displayFormat } = this.props
        const { chartConfig: { height } } = moreProps

        const config = chartConfig

        const origin = functor(originProp)
        const [x, y] = origin(width, height)
        const [ox, oy] = config.origin

        return (
            <g
                transform={`translate(${ ox + x }, ${ oy + y })`}
                className={className}
            >

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
    displayFormat: PropTypes.func.isRequired,
    origin: PropTypes.array.isRequired,
    onClick: PropTypes.func,
    fontFamily: PropTypes.string,
    fontSize: PropTypes.number,
    width: PropTypes.number,
    calculators: PropTypes.array.isRequired,
    forDataSeries: PropTypes.arrayOf(PropTypes.number),
}

ChartTooltip.defaultProps = {
    className: 'react-stockcharts-toottip react-stockcharts-moving-average-tooltip',
    displayFormat: format('.2f'),
    origin: [0, 10],
    width: 65,
}

export default ChartTooltip
