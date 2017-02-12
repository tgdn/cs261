import React from 'react'
import { Link } from 'react-router'

export default function Header() {
    return (
        <div class="luxbar luxbar-fixed">
            <input
                type="checkbox"
                id="luxbar-checkbox"
            />
            <div class="luxbar-menu luxbar-menu-dark luxbar-menu-black-force">
                <ul class="luxbar-navigation">
                    <li class="luxbar-header">
                        <Link
                            class="luxbar-brand"
                            to="/"
                        >
                            Purple
                        </Link>
                        <label
                            class="luxbar-hamburger luxbar-hamburger-spin"
                            htmlFor="luxbar-checkbox"
                        >
                            <span />
                        </label>
                    </li>
                    {/*
                    <li class="luxbar-item active"><a href="#">Home</a></li>
                    */}
                </ul>
            </div>
        </div>
    )
}
