/* eslint-disable react/forbid-prop-types */

import React, { PropTypes } from 'react'
import { Segment } from 'semantic-ui-react'
import SymbolChart from './symbolchart'

class SymbolDashboard extends React.Component {
    render() {
        const { symbol, trades } = this.props
        return (
            <div>
                <Segment inverted attached='bottom'>
                    <SymbolChart symbol={symbol} trades={trades} />
                </Segment>
            </div>
        )
    }
}

SymbolDashboard.propTypes = {
    symbol: PropTypes.string,
    trades: PropTypes.array,
    loadingError: PropTypes.bool,
}

SymbolDashboard.defaultProps = {
    symbol: '',
    trades: [],
    loadingError: false,
}

export default SymbolDashboard
