import React from 'react'
import { Link } from 'react-router'
import {
    Menu,
    Sidebar,
    Input
} from 'semantic-ui-react'

import map from 'lodash/map'
import filter from 'lodash/filter'

class AlertsSidebar extends React.Component {
    constructor(props) {
        super(props)
        this.handleInputChange = this.handleInputChange.bind(this)
        this.state = {
            searchTerm: '',
            alerts: this.props.alerts
        }
    }

    componentWillReceiveProps(newProps) {
        this.setState({ alerts: newProps.alerts })
    }

    handleInputChange(e) {
        const searchTerm = e.target.value
        this.setState({ searchTerm }, () => {
            if (searchTerm) {
                this.fetchSearch()
            } else {
                /* set back to initial */
                this.setState({
                    alerts: this.props.alerts
                })
            }
        })
    }

    fetchSearch() {
        const term = this.state.searchTerm
        if (term) {
            fetch('/api/alerts/search/', { // eslint-disable-line
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ term })
            })
            .then((res) => {
                if (res.status >= 200 && res.status < 300) {
                    return res.json()
                }
                const err = new Error(res.statusText)
                err.response = res
                throw err
            })
            .then((res) => {
                this.setState({
                    alerts: res.alerts
                })
            })
            .catch((err) => {
                this.setState({
                    alerts: []
                })
                throw err
            })
        }
    }

    render() {
        const { alertid = -1 } = this.props.params
        const { alerts } = this.state
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
                        placeholder='Alerts...'
                        icon='search'
                        value={this.state.searchTerm}
                        onChange={this.handleInputChange}
                    />
                </Menu.Item>
                {map(alerts, alert => (
                    <Menu.Item
                        key={alert.id}
                        as={Link}
                        active={alertid === alert.id}
                        to={`/flagged/${alert.id}`}
                    >
                        {alert.description}
                    </Menu.Item>
                ))}
            </Sidebar>
        )
    }
}

AlertsSidebar.propTypes = {
    horizon: React.PropTypes.any, // eslint-disable-line
    params: React.PropTypes.object, // eslint-disable-line
    alerts: React.PropTypes.array, // eslint-disable-line
}

AlertsSidebar.defaultProps = {
    horizon: null,
    params: {},
    alerts: [],
}

export default AlertsSidebar
