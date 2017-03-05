/* eslint-disable react/forbid-prop-types */

import React, { PropTypes } from 'react'
import { Segment, Header } from 'semantic-ui-react'
import SymbolChart from './symbolchart'

class SymbolDashboard extends React.Component {
    render() {
        const { symbol, trades } = this.props
        return (
            <div>
                <Segment inverted attached='bottom'>
                    {trades.length === 0 ? (
                        <Header inverted as='h2'>No data to work with</Header>
                    ) : (
                        <SymbolChart symbol={symbol} trades={trades} />
                    )}
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
