/* eslint-disable react/forbid-prop-types */

import React, { PropTypes } from 'react'

class SymbolDashboard extends React.Component {
    render() {
        return (
            <div>
                <div>
                    Chart
                </div>
                <div>
                    info
                </div>
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
