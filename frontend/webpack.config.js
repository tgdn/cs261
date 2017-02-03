const path = require('path');

module.exports = {
    entry: [
        'babel-polyfill',
        './src/index.js'
    ],
    output: {
        path: __dirname,
        publicPath: '/',
        filename: './dist/bundle.js'
    },
    module: {
        rules: [
            {
                enforce: 'pre',
                test: /\.(js|jsx)?$/,
                loader: 'eslint-loader',
                include: path.resolve(__dirname, 'src')
            },
            {
                exclude: /node_modules/,
                loader: 'babel-loader',
                query: {
                    plugins: ['transform-runtime'],
                    presets: ['es2015', 'stage-0', 'react'],
                }
            }
        ]
    },
    resolve: {
        extensions: ['.js', '.jsx']
    },
    devServer: {
        historyApiFallback: true,
        contentBase: './'
    }
};
