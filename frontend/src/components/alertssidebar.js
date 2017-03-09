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
            searching: false,
            alerts: this.props.alerts
        }
    }

    componentWillReceiveProps(newProps) {
        if (!this.state.searchTerm) {
            this.setState({ alerts: newProps.alerts })
        }
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
            this.setState({ searching: true })
            fetch('/api/alerts/search/', { // eslint-disable-line
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ term })
            })
            .then((res) => {
                this.setState({ searching: false })
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
                    alerts: [],
                    searching: false
                })
                throw err
            })
        }
    }

    render() {
        const { alertid = -1 } = this.props.params
        const { alerts, searching } = this.state
        return (
            <Sidebar
                as={Menu}
                width='thin'
                visible
                vertical
                inverted
                className='alertsidebar'
            >
                <Menu.Item>
                    <Input
                        inverted
                        placeholder='Alerts...'
                        icon='search'
                        value={this.state.searchTerm}
                        onChange={this.handleInputChange}
                        loading={searching}
                    />
                </Menu.Item>
                {map(alerts, alert => (
                    <Menu.Item
                        key={alert.id}
                        as={Link}
                        active={alertid === alert.id}
                        to={`/flagged/${alert.id}`}
                        className={`alert-item-${alert.severity}`}
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
