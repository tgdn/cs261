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
    state = {
        searchTerm: '',
    }

    render() {
        const { alertid = -1 } = this.props.params
        const alerts = this.props.alerts
        return (
            <Sidebar
                as={Menu}
                width='thin'
                visible
                vertical
                inverted
                className='symbolsidebar'
            >
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
    params: React.PropTypes.object, // eslint-disable-line
    alerts: React.PropTypes.array, // eslint-disable-line
}

AlertsSidebar.defaultProps = {
    params: {},
    alerts: [],
}

export default AlertsSidebar
