import App from './components/app'
import HomePage from './components/pages/home'
import SymbolPage from './components/pages/symbol'
import FlaggedPage from './components/pages/flagged'
import AlertPage from './components/pages/alert'
import PageDoesNotExist from './components/pages/404'

export default [
    {
        path: '/',
        component: App,
        indexRoute: {
            component: HomePage,
        },
        childRoutes: [
            {
                path: 'flagged',
                component: FlaggedPage,
                childRoutes: [
                    {
                        path: ':alertid',
                        component: AlertPage,
                    }
                ]
            },
            {
                path: ':symbol',
                component: SymbolPage,
            },
            {
                path: '*',
                component: PageDoesNotExist,
            }
        ]
    }
]
