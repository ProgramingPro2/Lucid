import * as THREE from "three";

const CLUSTER_COUNT = 12;

export function clusterColor(cluster: number): THREE.Color {
  const hue = (cluster % CLUSTER_COUNT) / CLUSTER_COUNT;
  return new THREE.Color().setHSL(hue, 0.72, 0.58);
}

export function clusterHex(cluster: number): string {
  return `#${clusterColor(cluster).getHexString()}`;
}

export { CLUSTER_COUNT };
