<template>
  <v-app id="inspire">
    <v-main>
      <catalog />
    </v-main>
  </v-app>
</template>

<script>
import Catalog from './components/Catalog';

export default {
  name: 'app',
  components: { Catalog }
};

</script>
