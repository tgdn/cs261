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
    alerts: React.PropTypes.array, // eslint-disable-line
}

AlertsSidebar.defaultProps = {
    alerts: [],
}

export default AlertsSidebar
