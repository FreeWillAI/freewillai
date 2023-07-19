// SPDX-License-Identifier: OTHER
pragma solidity ^0.8.9;

contract Utils{
    function hash(string memory key) public pure returns (bytes32) {
        return keccak256(abi.encodePacked(key));
    }
    
    function hash2(string memory key1, string memory key2) public pure returns (bytes32) {
        return keccak256(abi.encodePacked(key1, key2));
    }

    function equalStrings(string memory s1, string memory s2) public pure returns (bool){  
        return keccak256(abi.encodePacked(s1)) == keccak256(abi.encodePacked(s2));
    }

    function isEmptyString(string memory str) public pure returns (bool) {
        return bytes(str).length == 0;
    }
}
