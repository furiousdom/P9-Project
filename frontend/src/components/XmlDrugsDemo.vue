<template>
    <Panel header="Drugs" style="height: 100%">
        <div v-for="({ id, name}) in xdrugs" :key=id>
            <p>{{ name }}</p>
        </div>
    </Panel>
</template>

<script>
	import xdrugApi from '../service/xdrug';

	export default {
        data: () => ({ xdrugs: null }),
		async mounted() {
            const { data } = await xdrugApi.fetch();
            this.xdrugs = data;
		},
	}
</script>
