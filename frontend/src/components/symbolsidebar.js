import React from 'react'
import { Link } from 'react-router'
import {
    Menu,
    Sidebar,
    Input
} from 'semantic-ui-react'

import map from 'lodash/map'
import filter from 'lodash/filter'

const searchSymbols = (list, term) => {
    if (term) {
        return filter(list, symbol => (
            symbol.name.toLowerCase().search(term.toLowerCase()) !== -1
        ))
    }
    return list
}

class SymbolSidebar extends React.Component {
    state = {
        searchTerm: '',
    }

    handleInputChange = (e) => { this.setState({ searchTerm: e.target.value }) }

    render() {
        const symbolsRaw = this.props.symbols
        const symbols = searchSymbols(symbolsRaw, this.state.searchTerm)
        return (
            <Sidebar
                as={Menu}
                width='thin'
                visible
                vertical
                inverted
                className='symbolsidebar'
            >
                <Menu.Item>
                    <Input
                        inverted
                        placeholder='Symbols...'
                        icon='search'
                        value={this.state.searchTerm}
                        onChange={this.handleInputChange}
                    />
                </Menu.Item>
                {map(symbols, symbol => (
                    <Menu.Item
                        key={symbol.name}
                        as={Link}
                        to={`/${symbol.name}`}
                    >
                        {symbol.name}
                    </Menu.Item>
                ))}
            </Sidebar>
        )
    }
}

SymbolSidebar.propTypes = {
    symbols: React.PropTypes.array, // eslint-disable-line
}

SymbolSidebar.defaultProps = {
    symbols: [],
}

export default SymbolSidebar
